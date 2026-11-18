## PW1 lab A CSPC for physcs and chemistrycspc my cource work repository

- loop 2.0220 s
- numpy 0.0004 s
- speedup 5708.25x
- all 3 tests passed
- at the end the numpy much faster
- create the environment
- branch work started

##PW1 lab B : data,plotting and automation

-what was built :

- observed decay data and compared it with the analytical decay law.
- the observed data follows the same decreasing trend as the analytical decay law
- the Snakefile creates figure.png from decay_observed.csv using plot.py  

##PW2 lab A : motion from tracking data

-what was done :

- c alculated velocity and acceleration from noisy data
- integrated acceleration to recover velocity and position
- mean acceleration: -8.58 m/s2 standard deviation: 28.72 m/s2
- saved the three plots as a motion.png

##PW5 how far away was the earthquake

- the p wave arrived at 108.42 s and the s wave at 144.08 s
- using the p wave velocity of 6.2 km/s and the s wave velocity of 3.6 km/s:
- distance to the earthquake: approximately 306.3 km
- uncertainty: approximately ±8.6 km
- estimated origin time: 59.0 s
- using the arrival times instead of the maximum shaking gives a better estimate of the earthquake distance
- in step 7 the spectrogram shows how the frequency content changes over time
- before 108 s, the signal is mostly background noise
- the P-wave appears at approximately 108 s
- the stronger S-wave arrives at approximately 144 s and produces more intense energy
- later surface waves continue for a longer time and are mainly associated with lower frequencies
- the spectrogram shows when different frequencies appear in the seismic record
- in step 8 the magnitude part the max horizontal ground velocity i mean pgv was calculated using the raw N and E channels
- PGV = 1.5105 cm/s
- distance R = 306.3 km
- estimated magnitude M = 6.664
- the distance correction is important because ground motion becomes weaker as the distance from the earthquake increases. Ignoring distance would underestimate the magnitude.
- so an increase of one magnitude unit to a PGV increase by a factor of approximately 3.55 at the same distance