


def KWH2BTU(kwh):
    return kwh * 3412.142
def BTU2KWH(btu):
    return btu / 3412.142
def SQM2SQF(sqm):
    return sqm * 10.7639
def SQF2SQM(sqf):
    return sqf / 10.7639
def THM2BTU(thm):
    return thm * 99976.1
def BTU2THM(btu):
    return btu / 99976.1


def parse_climate_code(code):
    num = int(eval(code[0]))
    ltr = code[1:]
    return num, ltr