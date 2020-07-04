Software to analyse emission lines in an plasma with the goal to identify the elements within the plasma.

Two seperate ways of processing the plasma spectrum are implemented. After processing,
the resulting data is classified via a SVM chain-classifier.

1)

The mean of an interval over an emission line and the mean of an ajacent background radiation intervall are calculated.
The mean of the background is subtracted from the emission line mean. This is performed for pre defined
intervals of multiple emission lines of the relevant elements. The resulting corrected mean is used as the line intensity.

2)

The background is approximated by a spline fit through the local minima in between the emission lines.
The approximate background is then substracted from the spectrum.
Neighbouring emission lines are the fitted with overlapping Voight profiles for increased accuracy.
The same emission lines as in approch 1) are used here as well. The sum over the fitted Voigt profile is used as th line intensity.


The line intensities calculated after 1) and 2) are used as the input data for the SVM chain-cassefieres in:
