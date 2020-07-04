# Analysis_of_plasma_spectrums

Software to analyse emission lines in an plasma with the goal to identify the elements within the plasma.

Two seperate ways of processing the plasma spectrum are implemented. After processing,
the resulting data is classified via a SVM chain-classifier.

1)  2020_02_21_ROI_Auswertung_Final

The mean of an interval over an emission line and the mean of an ajacent background radiation intervall are calculated.
The mean of the background is subtracted from the emission line mean. This is performed for pre defined
intervals of multiple emission lines of the relevant elements. The resulting corrected mean is used as the line intensity.

2)  2020_02_21_CurveFit_Auswertung_Final

The background is approximated by a spline fit through the local minima in between the emission lines.
The approximate background is then substracted from the spectrum.
Neighbouring emission lines are the fitted with overlapping Voight profiles for increased accuracy.
The same emission lines as in approch 1) are used here as well. The sum over the fitted Voigt profile is used as the
line intensity. The function uses multiprocessing Pool to speed up the processing.


The line intensities calculated after 1) and 2) are used as the input data for the SVM chain-cassefieres.
It can be expected in the samples that multiple elements are in the plasma.
This means that the classification problem is a multilabel multiclass one.
the classification of the two approches are done in 2020_02_21_ROI_MLTSVM.


Additionnaly this repository containes the ROI_picker. A program implementet in TKinter that displayes the start and end boundaries
of the ROIs and their backgrounds. This is displayed over a measurend spectrum. The programm lets the user edit, add and delete ROIs
with a GUI and mouse clicks.