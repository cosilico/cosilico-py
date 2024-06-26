from pint import UnitRegistry

ureg = UnitRegistry()

def to_microns_per_pixel(resolution, resolution_unit):
    converted = (resolution * ureg(resolution_unit)).to('micron')
    return converted.magnitude