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
albedo_shortname = "albs"
surf_pres_short = "ps"
albedo_metadata = dict(
    long_name="Surface Albedo", units="1.0", standard_name="surface_albedo"
)
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


def compute_albedo(ds, swdown="rsds", swup="rsus"):
    """compute surface albedo from upwelling and downwelling shortwave radiation rsus/rsds"""

    albedo = xr.where(ds[swdown] > 0.0, ds[swup] / ds[swdown], CMOR_MISSING_VALUE)

    # copy attributes from input field
    albedo.attrs = ds[swdown].attrs.copy()
    # remove interp_method for fregrid
    if "interp_method" in albedo.attrs:
        albedo.attrs.pop("interp_method")
    # update name and units
    albedo.attrs.update(albedo_metadata)
    # add comment on how albedo was computed
    albedo.attrs.update({"comment": f"{swup}/{swdown}"})

    return albedo


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


def refine_tracers(ds, refined, new_vars_output, verbose=False):
    """compute additional tracers"""

    all_vars = set(ds.variables)

    if set(["emipoa", "chepsoa"]).issubset(all_vars):
        new_vars_output = True
        refined["emioa"] = ds["emipoa"] + ds["chepsoa"]
        refined["emioa"].attrs = ds["emipoa"].attrs.copy()
        refined["emioa"].attrs.update(
            {
                "long_name": "Rate of Emission and Production of Dry Aerosol Total Organic Matter",
                "units": "kg m-2 s-1",
                "comment": "emipoa+chepsoa",
                "standard_name": "tendency_of_atmosphere_mass_content_of_particulate_organic_matter_dry_aerosol_due_to_net_chemical_production_and_emission",
            }
        )

    if set(["emiapoa", "chepasoa"]).issubset(all_vars):
        new_vars_output = True
        refined["emiaoa"] = ds["emiapoa"] + ds["chepasoa"]
        refined["emiaoa"].attrs = ds["emiapoa"].attrs.copy()
        refined["emiaoa"].attrs.update(
            {
                "long_name": "total emission of anthropogenic organic aerosol",
                "units": "kg m-2 s-1",
                "comment": "emiapoa+chepasoa",
                "standard_name": "tendency_of_atmosphere_mass_content_of_particulate_organic_matter_dry_aerosol_particles_due_to_net_chemical_production_and_emission",
            }
        )

    if set(["drypoa", "drysoa"]).issubset(all_vars):
        new_vars_output = True
        refined["dryoa"] = ds["drypoa"] + ds["drysoa"]
        refined["dryoa"].attrs = ds["drypoa"].attrs.copy()
        refined["dryoa"].attrs.update(
            {
                "long_name": "Dry Deposition Rate of Dry Aerosol Total Organic Matter",
                "units": "kg m-2 s-1",
                "comment": "drypoa+drysoa",
                "standard_name": "tendency_of_atmosphere_mass_content_of_particulate_organic_matter_dry_aerosol_due_to_dry_deposition",
            }
        )

    if set(["wetpoa", "wetsoa"]).issubset(all_vars):
        new_vars_output = True
        refined["wetoa"] = ds["wetpoa"] + ds["wetsoa"]
        refined["wetoa"].attrs = ds["wetpoa"].attrs.copy()
        refined["wetoa"].attrs.update(
            {
                "long_name": "Wet Deposition Rate of Dry Aerosol Total Organic Matter",
                "units": "kg m-2 s-1",
                "comment": "wetpoa+wetsoa",
                "standard_name": "tendency_of_atmosphere_mass_content_of_particulate_organic_matter_dry_aerosol_due_to_wet_deposition",
            }
        )

    if set(["pso4_aq_kg_m2_s", "pso4_aq_so2_reevap_ls"]).issubset(all_vars):
        new_vars_output = True
        refined["cheaqpso4"] = ds["pso4_aq_kg_m2_s"] + ds["pso4_aq_so2_reevap_ls"]
        refined["cheaqpso4"].attrs = ds["pso4_aq_kg_m2_s"].attrs.copy()
        refined["cheaqpso4"].attrs.update(
            {
                "long_name": "Aqueous-phase Production Rate of Sulfate Aerosol",
                "units": "kg m-2 s-1",
                "comment": "pso4_aq_kg_m2_s+pso4_aq_so2_reevap_ls",
                "standard_name": "tendency_of_atmosphere_mass_content_of_sulfate_dry_aerosol_particles_due_to_aqueous_phase_net_chemical_production",
            }
        )

    if set(["eminox_woL", "eminox_lght"]).issubset(all_vars):
        new_vars_output = True
        refined["eminox"] = ds["eminox_woL"] + ds["eminox_lght"]
        refined["eminox"].attrs = ds["eminox_woL"].attrs.copy()
        refined["eminox"].attrs.update(
            {
                "long_name": "Total Emission Rate of NOx",
                "units": "kg m-2 s-1",
                "comment": "eminox_woL+eminox_lght",
                "standard_name": "tendency_of_atmosphere_mass_content_of_nox_expressed_as_nitrogen_due_to_emission",
            }
        )

    if set(["emiisop_woB", "emiisop_biogenic"]).issubset(all_vars):
        new_vars_output = True
        refined["emiisop"] = ds["emiisop_woB"] + ds["emiisop_biogenic"]
        refined["emiisop"].attrs = ds["emiisop_woB"].attrs.copy()
        refined["emiisop"].attrs.update(
            {
                "long_name": "Total Emission Rate of Isoprene",
                "units": "kg m-2 s-1",
                "comment": "emiisop_woB+emiisop_biogenic",
                "standard_name": "tendency_of_atmosphere_mass_content_of_isoprene_due_to_emission",
            }
        )

    if set(["eminh3_woOCN", "nh3_mol_flux_atm0"]).issubset(all_vars):
        new_vars_output = True
        refined["eminh3"] = ds["eminh3_woOCN"] + 0.017 * ds["nh3_mol_flux_atm0"]
        refined["eminh3"].attrs = ds["eminh3_woOCN"].attrs.copy()
        refined["eminh3"].attrs.update(
            {
                "long_name": "Total Emission Rate of NH3",
                "units": "kg m-2 s-1",
                "comment": "eminh3_woOCN+0.017*nh3_mol_flux_atm0",
                "standard_name": "tendency_of_atmosphere_mass_content_of_ammonia_due_to_emission",
            }
        )

    if set(["drynh3_woOCN", "nh3_mol_flux_atm0", "nh3_mol_flux"]).issubset(all_vars):
        new_vars_output = True
        refined["drynh3"] = (
            ds["drynh3_woOCN"]
            + 0.017 * ds["nh3_mol_flux_atm0"]
            - 0.017 * ds["nh3_mol_flux"]
        )
        refined["drynh3"].attrs = ds["drynh3_woOCN"].attrs.copy()
        refined["drynh3"].attrs.update(
            {
                "long_name": "Dry Deposition Rate of NH3",
                "units": "kg m-2 s-1",
                "comment": "drynh3_woOCN+0.017*nh3_mol_flux_atm0-0.017*nh3_mol_flux",
                "standard_name": "minus_tendency_of_atmosphere_mass_content_of_ammonia_due_to_dry_deposition",
            }
        )

    if set(
        ["dust1_flux", "dust2_flux", "dust3_flux", "dust4_flux", "dust5_flux"]
    ).issubset(all_vars):
        new_vars_output = True
        refined["emidust"] = (
            ds["dust1_flux"]
            + ds["dust2_flux"]
            + ds["dust3_flux"]
            + ds["dust4_flux"]
            + ds["dust5_flux"]
        )
        refined["emidust"] = refined["emidust"].clip(vmin=0.0)
        refined["emidust"].attrs = ds["dust1_flux"].attrs.copy()
        refined["emidust"].attrs.update(
            {
                "long_name": "Total Emission Rate of Dust",
                "units": "kg m-2 s-1",
                "comment": "dust1_flux+dust2_flux+dust3_flux+dust4_flux+dust5_flux",
                "standard_name": "tendency_of_atmosphere_mass_content_of_dust_dry_aerosol_particles_due_to_emission",
            }
        )

    return refined, new_vars_output


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
        "-d",
        "--shortwave_down",
        type=str,
        required=False,
        default="rsds",
        help="name of downwards shortwave radiation",
    )
    parser.add_argument(
        "-u",
        "--shortwave_up",
        type=str,
        required=False,
        default="rsus",
        help="name of upwards shortwave radiation",
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
