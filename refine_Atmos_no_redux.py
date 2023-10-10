#!/usr/bin/env python

# This script contains the refineDiags that produce data at the same
# frequency as the input data (no reduction) such as surface albedo,
# masking fields,...
# It can accept any file and will only compute refineDiags in fields
# are present.

import argparse
import os
import netCDF4 as nc
import xarray as xr

def run():
    args = parse_args()

    # Error if outfile exists
    if os.path.exists(args.outfile):
        raise Exception(f"ERROR: Output file '{args.outfile}' already exists")

    ds_in = xr.open_dataset(args.infile)

    # Exit with message if "ps" not available
    if "ps" not in list(ds_in.variables):
        print(f"WARNING: Input file '{args.infile}' does not contain surface pressure, so exiting")
        return None

    # The trigger for atmos masking is a variable attribute "needs_atmos_masking = True".
    # In the future this will be set within the model, but for now and testing,
    # we'll add the attribute for variables that end with "_unmsk".
    # At the same time, strip the "_unmsk" from the variable name.
    ds_in = preprocess(ds_in)

    ds_out = xr.Dataset()

    # Process all variables with attribute "needs_atmos_masking = True"
    for var in list(ds_in.variables):
        if 'needs_atmos_masking' in ds_in[var].attrs:
            del ds_in[var].attrs['needs_atmos_masking']
            ds_out[var] = mask_field_above_surface_pressure(ds_in, var)
        else:
            continue

    # Write the output file if anything was done
    if ds_out.variables:
        print(f"Modifying variables '{list(ds_out.variables)}', appending into new file '{args.outfile}'")
        write_dataset(ds_out, ds_in, args.outfile)
    else:
        print(f"No variables modified, so not writing output file '{args.outfile}'")
    return None


def preprocess(ds):
    """add needs_atmos_masking attribute if var ends with _unmsk"""

    for var in list(ds.variables):
        if var.endswith('_unmsk'):
            ds[var].attrs['needs_atmos_masking'] = True
            newvar = var.replace("_unmsk", "")
            ds = ds.rename_vars({var: newvar})

    return ds


def mask_field_above_surface_pressure(ds, var):
    """mask data with pressure larger than surface pressure"""

    plev = pressure_coordinate(ds, var)

    # broadcast pressure coordinate and surface pressure to
    # the dimensions of the variable to mask
    plev_extended, _ = xr.broadcast(plev, ds[var])
    ps_extended, _ = xr.broadcast(ds["ps"], ds[var])
    # masking do not need looping
    masked = xr.where(plev_extended > ps_extended, 1.0e20, ds[var])
    # copy attributes and transpose dims like the original array
    masked.attrs = ds[var].attrs.copy()
    masked = masked.transpose(*ds[var].dims)

    print(f"Processed {var}")

    return masked


def pressure_coordinate(ds, varname, verbose=False):
    """check if dataArray has pressure coordinate fitting requirements
    and return it"""

    pressure_coord = None

    for dim in list(ds[varname].dims):
        if dim in list(ds.variables):  # dim needs to have values in file
            if ds[dim].attrs["long_name"] == "pressure":
                pressure_coord = ds[dim]
            elif ("coordinates" in ds.attrs) and (ds[dim].attrs["units"] == "Pa"):
                pressure_coord = ds[dim]

    return pressure_coord


def write_dataset(ds, template, outfile):
    """prepare the dataset and dump into netcdf file"""

    # copy global attributes
    ds.attrs = template.attrs.copy()

    # copy all variables and their attributes
    # except those already processed
    for var in list(template.variables):
        if var in list(ds.variables):
            continue
        else:
            ds[var] = template[var]
            ds[var].attrs = template[var].attrs.copy()

    ds.to_netcdf(outfile, unlimited_dims="time")

    return None


def set_netcdf_encoding(ds, pressure_vars):
    """set preferred options for netcdf encoding"""

    all_vars = list(ds.variables)
    encoding = {}

    for var in do_not_encode_vars + pressure_vars:
        if var in all_vars:
            encoding.update({var: dict(_FillValue=None)})

    return encoding


def post_write(filename, ds, var_with_bounds, bounds_variables):
    """fix a posteriori attributes that xarray.to_netcdf
    did not do properly using low level netcdf lib"""

    f = nc.Dataset(filename, "a")

    for var, bndvar in zip(var_with_bounds, bounds_variables):
        f.variables[var].setncattr("bounds", bndvar)

    f.close()

    return None


def parse_args():
    """parse command line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input file")
    parser.add_argument(
        "-o", "--outfile", type=str, required=True, help="Output file name"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        help="Print detailed output",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        required=False,
        default="NETCDF3_64BIT",
        help="netcdf format for output file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run()
