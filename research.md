---
layout: researchpage
title: Research
subtitle: Current Projects
---

**Clustering Analysis of Autumn Weather Regimes in the Northeast U.S.** 
- _Funded by NASA NESSF Grant NNX15AN91H S03, NOAA NA20OAR4310424, and NSF AGS-1623912_
- **Supervisor: Dr. Mathew Barlow**
- ![image](assets/img/fall_season.png)
- Identified typical autumn circulation patterns across the Northeast U.S. using a k-means clustering algorithm coded in MATLAB.
- Analyzed seasonality of the circulation patterns, confirming previous observations from other studies that summer is lasting longer in the year and winter starting later.
- [Read the Paper!](https://doi.org/10.1175/JCLI-D-20-0243.1)
- [View the code!](https://github.com/Daviology38/Analysis_from_Coe_et_al_2021_Fall_Season)

**Analyzing the Role of Integrated Vapor Transport and Diabatic Heating on Extreme Precipitation Events in the Northeast U.S.**
- _Funded by NOAA NA20OAR4310424, and NSF AGS-1623912_
- **Supervisor: Dr. Mathew Barlow**
- Modelled 20 extreme-precipitation events in the Northeast U.S. using the WRF-ARW that were associated with Warm Conveyor Belts and Moisture Transport.
- Configured WRF-ARW to run with and without diabatic heating terms using a two-way nested model of resolutions 27km and 3km.
- Coded python scripts to plot and analyze the output, determining that diabatic heating played a significant role in Integrated Vapor Transport, its absence causing weakening of extreme precipitation events, with some events not producing enough precipitation to cross the threshold of extreme precipitation.  
- Lead author on paper in development.

**Spring Season Circulation Evolution and Relation to Extreme Events in the Northeast U.S.**
- _Funded by NASA NESSF Grant NNX15AN91H S03, NOAA NA20OAR4310424, and NSF AGS-1623912_
- **Supervisor: Dr. Mathew Barlow**
- ![image](assets/img/spring_season.png)
- Identified spring season circulation patterns using K-means clustering and performed analysis on seasonal evolution using them.
- Enhanced understanding of extreme Precipitation, Heat wave, and Drought events during the spring season in the Northeast U.S. by investigating their relation to the underlying seasonal circulation patterns.
- Lead author on paper in preparation.

**Understanding Winter Season Onset and Withdrawal Timing in the Northeast U.S. Using Circulation Patterns and Deep Learning**
- **Supervisor: Dr. Mathew Barlow**
- ![image](assets/img/neural_network_output.PNG)
- Developed a Siamese Neural Network model to perform image matching on small datasets.
- Matched out of season dates to typical spring and fall circulation patterns to determine typical onset and withdrawal of winter.
- Lead author on paper in development.

**A Climatology of Snow Squalls in Southern New England 1994 - 2018**
- **Supervisor: Dr. Frank Colby**
- Coded FORTRAN and Python scripts to download METAR station data and determine dates, times, and locations of snow squall events based on given observations. Created a dataset of NEXRAD files for the dates and times of squalls based on proximity to radar sites KBOX and KENX.
- Reconfigured existing Python script (TINT) for cell tracking to work with the NEXRAD dataset of snow squall events and output the average direction of cell movement as both a .csv file and plotted on a radial plot of cardinal directions.
- Snow squalls were grouped based on average direction of movement, resulting in 4 main types of squalls that varied in orientation and direction of movement based on associated fronts and available convection (either or both CAPE and slantwise). 
- Co-author on paper in review at Monthly Weather Review.