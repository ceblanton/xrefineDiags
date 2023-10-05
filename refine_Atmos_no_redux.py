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


CMOR_MISSING_VALUE = 1.0e20
extra_time_variables = ["time_bnds", "average_T1", "average_T2", "average_DT"]
do_not_encode_vars = ["nv", "grid_xt", "grid_yt", "time", "lev"]
grid_vars = ["grid_xt", "grid_yt", "ap", "b", "ap_bnds", "b_bnds", "lev", "lev_bnds"]
unaccepted_variables_for_masking = ["cll", "clm", "clh"]
surf_pres_short = "ps"
pkgname = "xrefineDiags"
scriptname = "refine_Atmos_no_redux.py"
pro = f"{pkgname}/{scriptname}"


def run():
    """run the refineDiags"""

    # --- parse command line arguemnts
    args = parse_args()
    verbose = args.verbose

    # --- open file
    if verbose:
        print(f"{pro}: Opening input file {args.infile}")

    ds = xr.open_dataset(args.infile, decode_cf=False)

    # --- create output dataset
    if os.path.exists(args.outfile):
        if verbose:
            print(f"{pro}: Opening existing file {args.outfile}")

        refined = xr.open_dataset(args.outfile, decode_cf=False)
        refined.load()
    else:
        if verbose:
            print(f"{pro}: Creating new dataset")

        refined = xr.Dataset()

    # we haven't created any new variable yet
    new_vars_output = False

    # --- mask variables with surface pressure
    refined, new_vars_output, pressure_vars = mask_above_surface_pressure(
        ds, refined, new_vars_output, surf_pres_short=surf_pres_short, verbose=verbose
    )

    # --- write dataset to file
    if verbose and new_vars_output:
        print(
            f"{pro}: writting variables {list(refined.variables)} into refined file {args.outfile} "
        )
    elif verbose and not new_vars_output:
        print(f"{pro}: no variables created, not writting refined file")

    if new_vars_output:
        write_dataset(refined, ds, pressure_vars, args)

    return None


def mask_above_surface_pressure(
    ds, refined, new_vars_output, surf_pres_short="ps", verbose=False
):
    """find fields with pressure coordinate and mask
    values of fields where p > surface pressure

    Args:
        ds (_type_): _description_
        out (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.
    """

    pressure_vars = []

    # surface pressure needs to be in the dataset
    if surf_pres_short in list(ds.variables):
        vars_to_process = list(ds.variables)
        # do not process surface pressure
        vars_to_process.remove(surf_pres_short)
        for var in vars_to_process:
            # find the pressure coordinate in dataset
            plev = pressure_coordinate(ds, var, verbose=verbose)
            # proceed if there is a coordinate pressure
            # but do not process the coordinate itself
            if (plev is not None) and (var != plev.name):
                pressure_vars.append(plev.name)
                new_vars_output = True
                refined[var] = mask_field_above_surface_pressure(
                    ds, var, plev, surf_press_short=surf_pres_short
                )
                refined[plev.name].attrs = ds[plev.name].attrs.copy()

    pressure_vars = list(set(pressure_vars))

    return refined, new_vars_output, pressure_vars


def mask_field_above_surface_pressure(ds, var, pressure_dim, surf_press_short="ps"):
    """mask data with pressure larger than surface pressure"""

    # broadcast pressure coordinate and surface pressure to
    # the dimensions of the variable to mask
    plev_extended, _ = xr.broadcast(pressure_dim, ds[var])
    ps_extended, _ = xr.broadcast(ds[surf_press_short], ds[var])
    # masking do not need looping
    masked = xr.where(plev_extended > ps_extended, CMOR_MISSING_VALUE, ds[var])
    # copy attributes and transpose dims like the original array
    masked.attrs = ds[var].attrs.copy()
    masked = masked.transpose(*ds[var].dims)

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

    # some variables need not to be masked
    if varname in unaccepted_variables_for_masking:
        pressure_coord = None

    if verbose:
        if pressure_coord is not None:
            print(f"{pro}: {varname} has pressure coords {pressure_coord.name}")
        else:
            print(f"{pro}: {varname} has no pressure coords")

    return pressure_coord


def write_dataset(ds, template, pressure_vars, args):
    """prepare the dataset and dump into netcdf file"""

    if len(ds.attrs) == 0:
        ds.attrs = template.attrs.copy()  # copy global attributes
    ds.attrs["filename"] = args.outfile

    # --- add proper grid attrs
    for var in grid_vars:
        if var in list(template.variables):
            ds[var] = template[var]
            ds[var].attrs = template[var].attrs.copy()

    # --- add extra time variables
    for var in extra_time_variables:
        if var in list(template.variables):
            ds[var] = template[var]
            ds[var].attrs = template[var].attrs.copy()

    # --- remove bounds in attributes since it messed the bnds var
    var_with_bounds = []
    bounds_variables = []
    for var in list(ds.variables):
        if "bounds" in ds[var].attrs:
            var_with_bounds.append(var)
            bounds_variables.append(ds[var].attrs.pop("bounds"))

    encoding = set_netcdf_encoding(ds, pressure_vars)

    ds.to_netcdf(
        args.outfile, format=args.format, encoding=encoding, unlimited_dims="time"
    )
    post_write(args.outfile, ds, var_with_bounds, bounds_variables)

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
    parser.add_argument("-t", "--tagfile", action="store_true", required=False, help="")
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
