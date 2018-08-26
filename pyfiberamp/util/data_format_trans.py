from pyfiberamp.helper_funcs import *

def txt2dat(txt_read_path, dat_write_path):
    f_dat = open(dat_write_path, "wb")
    with open(txt_read_path) as f_txt:
        for line in f_txt.readlines():
            line = line.rstrip("\n")
            wavelength, absorption = line.split("\t")
            wavelength = float(wavelength) * 1e9
            absorption = float(absorption)
            wline = "{0}\t{1}\n".format(wavelength, absorption)
            wline = wline.replace(".", ",")
            wline = wline.encode()
            f_dat.write(wline)
            print(wline)

    f_dat.close()


def test_txt2dat(dat_path):
    x = load_spectrum(dat_path)
    print(x.shape)
    print(x)


if __name__ == "__main__":
    # fpath_txt = "E:\Data\ROF\edfa_link\edfa_data\\absorption_parameters_display.txt"
    # fpath_dat = "G:\Win\Workspace\Python\pyfiberamp\pyfiberamp\spectroscopies\\fiber_spectra\\erbium_absorption_cross_sections.dat"
    fpath_txt = "E:\Data\ROF\edfa_link\edfa_data\\emission_parameters_display.txt"
    fpath_dat = "G:\Win\Workspace\Python\pyfiberamp\pyfiberamp\spectroscopies\\fiber_spectra\\erbium_emission_cross_sections.dat"
    txt2dat(fpath_txt, fpath_dat)
    test_txt2dat(fpath_dat)

