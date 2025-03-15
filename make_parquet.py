import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.cosmology import z_at_value

vel=False
vel=True


def simulator(seed, number_density, outfile):

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Cosmological parameters
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    # Number density in h^3 Mpc^-3


    # Maximum redshift
    z_max = 0.1

    # Calculate comoving volume up to z_max in Mpc^3
    volume_in_Mpc3 = 4/3 * np.pi * cosmo.comoving_distance(z_max).value**3

    # Convert to (h^-1 Mpc)^3 units
    h = cosmo.H0.value / 100
    volume_in_h_units = volume_in_Mpc3 * h**3

    # Calculate expected number of points
    n_points = int(number_density * volume_in_h_units)
    print(f"Creating {n_points} points in a volume of {volume_in_h_units:.2e} (h^-1 Mpc)^3")

    # Generate random points in spherical coordinates
    r_max = cosmo.comoving_distance(z_max).value
    r = r_max * np.cbrt(np.random.random(n_points))

    phi = 2 * np.pi * np.random.random(n_points)
    cos_theta = 2 * np.random.random(n_points) - 1
    theta = np.arccos(cos_theta)

    # Convert to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Calculate distance, RA, Dec
    distance = np.sqrt(x**2 + y**2 + z**2)

    ra_rad = np.arctan2(y, x)
    dec_rad = np.arcsin(z / distance)

    ra_deg = np.rad2deg(ra_rad) % 360
    dec_deg = np.rad2deg(dec_rad)

    # Calculate redshift from distance
    redshift = np.array([z_at_value(cosmo.comoving_distance, d * u.Mpc) for d in distance])

    # Create DataFrame
    df = pd.DataFrame({
        'ra': ra_rad,
        'dec': dec_rad,
        'zobs': redshift,
    })

    # Convert to Galactic coordinates
    coords = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree, frame='icrs')
    galactic = coords.galactic

    # Apply selection criteria
    mask = (dec_deg < -10) & (np.abs(galactic.b.degree) > 20)
    filtered_df = df[mask]

    print(f"Selected {len(filtered_df)} points after applying criteria")

    # Save to parquet file
    table = pa.Table.from_pandas(filtered_df)
    pq.write_table(table, outfile)

    print("Data saved to {}".format(outfile))

#velocity
simulator(42, 5e-4/5,'file_vel.parquet')

# density
simulator(43, 5e-4*10,'file_gal.parquet')
