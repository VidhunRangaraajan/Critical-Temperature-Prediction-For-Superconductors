# predict.py
# Description: 
# Author: Vidhun Rangaraajan J
# Website: https://www.vidhun.com
# Github: https://github.com/VidhunRangaraajan
# Repository: https://github.com/VidhunRangaraajan/Critical-Temperature-Prediction-For-Superconductors
# Requirements: 
# Usage: 
# Depends On: 
# Input Files: 
# Output Files: 
# Notes: 
# To Do: 

# Importing the predict_mode function from the "src/predict.py" module/file.
from src.predict import predict_mean

# Actual critical_temp = 72.85
sample1 = {
    "number_of_elements": 5.0, "mean_atomic_mass": 74.23165, "wtd_mean_atomic_mass": 51.6606684824903,
    "gmean_atomic_mass": 60.5152205498845, "wtd_gmean_atomic_mass": 35.257147052647,
    "entropy_atomic_mass": 1.45374713602643, "wtd_entropy_atomic_mass": 1.30424654802576,
    "range_atomic_mass": 121.3276, "wtd_range_atomic_mass": 21.2466536964981,
    "std_atomic_mass": 39.4458745649529, "wtd_std_atomic_mass": 44.2850029733093,
    "mean_fie": 816.36, "wtd_mean_fie": 1000.3233463035, "gmean_fie": 771.891022146416,
    "wtd_gmean_fie": 934.234826909982, "entropy_fie": 1.55250636371223, "wtd_entropy_fie": 0.908859369847376,
    "range_fie": 810.6, "wtd_range_fie": 698.217120622568, "std_fie": 282.395259167005, "wtd_std_fie": 342.583008316592,
    "mean_atomic_radius": 160.0, "wtd_mean_atomic_radius": 115.309338521401, "gmean_atomic_radius": 139.595298984601,
    "wtd_gmean_atomic_radius": 90.3383372923269, "entropy_atomic_radius": 1.50121758487734,
    "wtd_entropy_atomic_radius": 1.35288961146792, "range_atomic_radius": 205.0, "wtd_range_atomic_radius": 39.1011673151751,
    "std_atomic_radius": 69.9228145886591, "wtd_std_atomic_radius": 79.3657471419932, "mean_Density": 4816.6858,
    "wtd_mean_Density": 2983.36876653696, "gmean_Density": 1074.9034884109, "wtd_gmean_Density": 69.3512240318849,
    "entropy_Density": 1.32221306724433, "wtd_entropy_Density": 0.84127941768554, "range_Density": 8958.571,
    "wtd_range_Density": 2073.63512451362, "std_Density": 3084.72934185263, "wtd_std_Density": 3641.31034058742,
    "mean_ElectronAffinity": 62.09, "wtd_mean_ElectronAffinity": 108.549805447471, "gmean_ElectronAffinity": 26.1868243919949,
    "wtd_gmean_ElectronAffinity": 85.3067787704854, "entropy_ElectronAffinity": 1.13042757181707,
    "wtd_entropy_ElectronAffinity": 0.768001809658304, "range_ElectronAffinity": 141.0,
    "wtd_range_ElectronAffinity": 75.9601167315175, "std_ElectronAffinity": 57.6447430387195,
    "wtd_std_ElectronAffinity": 49.823649412886, "mean_FusionHeat": 8.0144, "wtd_mean_FusionHeat": 5.29781712062257,
    "gmean_FusionHeat": 4.54968458544814, "wtd_gmean_FusionHeat": 1.36370119441867, "entropy_FusionHeat": 1.38465705322041,
    "wtd_entropy_FusionHeat": 1.05978313193, "range_FusionHeat": 12.878, "wtd_range_FusionHeat": 3.01857976653696,
    "std_FusionHeat": 4.43821466808445, "wtd_std_FusionHeat": 5.64534639571786, "mean_ThermalConductivity": 111.005316,
    "wtd_mean_ThermalConductivity": 96.9791496498054, "gmean_ThermalConductivity": 13.1311731511324,
    "wtd_gmean_ThermalConductivity": 1.14226484634753, "entropy_ThermalConductivity": 0.785623847019654,
    "wtd_entropy_ThermalConductivity": 0.220831267648683, "range_ThermalConductivity": 399.97342,
    "wtd_range_ThermalConductivity": 92.5928347859922, "std_ThermalConductivity": 150.600199073704,
    "wtd_std_ThermalConductivity": 166.549633197962, "mean_Valence": 2.2, "wtd_mean_Valence": 2.07782101167315,
    "gmean_Valence": 2.1689435423954, "wtd_gmean_Valence": 2.06411360125221, "entropy_Valence": 1.59416669911802,
    "wtd_entropy_Valence": 1.21871868289682, "range_Valence": 1.0, "wtd_range_Valence": 1.06225680933852,
    "std_Valence": 0.4, "wtd_std_Valence": 0.267889719502856,
}

# Actual critical_temp = 7.85
sample2 = {
    "number_of_elements": 4.0, "mean_atomic_mass": 113.5267, "wtd_mean_atomic_mass": 74.02931,
    "gmean_atomic_mass": 75.1983513978218, "wtd_gmean_atomic_mass": 37.8311117954341,
    "entropy_atomic_mass": 1.15875211295482, "wtd_entropy_atomic_mass": 1.15852517210338,
    "range_atomic_mass": 164.1558, "wtd_range_atomic_mass": 29.58786, "std_atomic_mass": 65.8933425178143,
    "wtd_std_atomic_mass": 68.2154031060574, "mean_fie": 651.125, "wtd_mean_fie": 709.325,
    "gmean_fie": 641.554365936701, "wtd_gmean_fie": 699.345857667068, "entropy_fie": 1.3716130832406,
    "wtd_entropy_fie": 1.08021321323552, "range_fie": 287.2, "wtd_range_fie": 371.05, "std_fie": 111.491106708114,
    "wtd_std_fie": 112.523050416348, "mean_atomic_radius": 177.0, "wtd_mean_atomic_radius": 142.7,
    "gmean_atomic_radius": 166.007174190852, "wtd_gmean_atomic_radius": 131.027361469629,
    "entropy_atomic_radius": 1.33169534534802, "wtd_entropy_atomic_radius": 1.288336814112,
    "range_atomic_radius": 139.0, "wtd_range_atomic_radius": 33.2, "std_atomic_radius": 55.0045452667323,
    "wtd_std_atomic_radius": 57.5657015939179, "mean_Density": 8434.25, "wtd_mean_Density": 6744.0,
    "gmean_Density": 7218.38327858053, "wtd_gmean_Density": 5188.46164425588, "entropy_Density": 1.27131090803236,
    "wtd_entropy_Density": 1.20876405828916, "range_Density": 9910.0, "wtd_range_Density": 2639.2,
    "std_Density": 3659.22958933981, "wtd_std_Density": 4390.57661361238, "mean_ElectronAffinity": 58.5,
    "wtd_mean_ElectronAffinity": 52.675, "gmean_ElectronAffinity": 52.6579769737295, "wtd_gmean_ElectronAffinity": 45.2975423901722,
    "entropy_ElectronAffinity": 1.282662396304, "wtd_entropy_ElectronAffinity": 1.16959842895934,
    "range_ElectronAffinity": 74.6, "wtd_range_ElectronAffinity": 23.125, "std_ElectronAffinity": 27.2881842561941,
    "wtd_std_ElectronAffinity": 30.4628769980775, "mean_FusionHeat": 29.4, "wtd_mean_FusionHeat": 36.82,
    "gmean_FusionHeat": 27.3870163597243, "wtd_gmean_FusionHeat": 34.3080056546237, "entropy_FusionHeat": 1.31019528207519,
    "wtd_entropy_FusionHeat": 0.918976225358791, "range_FusionHeat": 30.1, "wtd_range_FusionHeat": 24.005,
    "std_FusionHeat": 12.0733176881916, "wtd_std_FusionHeat": 13.2668986579381, "mean_ThermalConductivity": 44.5,
    "wtd_mean_ThermalConductivity": 47.45, "gmean_ThermalConductivity": 29.6953920230386,
    "wtd_gmean_ThermalConductivity": 34.2851262535249, "entropy_ThermalConductivity": 0.976902780132916,
    "wtd_entropy_ThermalConductivity": 0.894901957291908, "range_ThermalConductivity": 105.0,
    "wtd_range_ThermalConductivity": 29.25, "std_ThermalConductivity": 43.8434715778758,
    "wtd_std_ThermalConductivity": 42.1360593791114, "mean_Valence": 3.75, "wtd_mean_Valence": 3.75,
    "gmean_Valence": 3.56762134500816, "wtd_gmean_Valence": 3.56762134500816, "entropy_Valence": 1.33217904021012,
    "wtd_entropy_Valence": 1.15500065269378, "range_Valence": 3.0, "wtd_range_Valence": 1.35,
    "std_Valence": 1.29903810567666, "wtd_std_Valence": 1.29903810567666,
}

print(predict_mean(sample1))  # expect mean prediction near 72.85
print(predict_mean(sample2))  # expect mean prediction near 7.85