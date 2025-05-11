# Delay_data
# Feature Description
| Feature Name        | Description                                                                 | Example                                    |
|---------------------|-----------------------------------------------------------------------------|--------------------------------------------|
| FL_DATE             | Flight date and time                                                         | "2024/1/1 0:00"                           |
| OP_CARRIER          | Operating airline code                                                       | "9E"                                       |
| OP_CARRIER_FL_NUM   | Operating carrier flight number                                              | "4814"                                     |
| ORIGIN              | Origin airport code                                                          | "JFK" (John F. Kennedy International Airport) |
| DEST                | Destination airport code                                                     | "DTW" (Detroit Metropolitan Airport)       |
| CRS_DEP_TIME        | Scheduled departure time                                                     | "2024/1/1 12:52"                           |
| DEP_TIME            | Actual departure time                                                       | "2024/1/1 12:47"                           |
| DEP_DELAY           | Departure delay in minutes                                                   | "-5" (5 minutes early)                     |
| TAXI_OUT            | Taxi-out time before takeoff in minutes                                      | "31"                                       |
| WHEELS_OFF          | Time when the plane's wheels leave the ground                                | "2024/1/1 13:18"                           |
| WHEELS_ON           | Time when the plane's wheels touch the ground at the destination              | "2024/1/1 14:42"                           |
| TAXI_IN             | Taxi-in time after landing in minutes                                        | "7"                                        |
| CRS_ARR_TIME        | Scheduled arrival time                                                       | "2024/1/1 15:08"                           |
| ARR_TIME            | Actual arrival time                                                         | "2024/1/1 14:49"                           |
| ARR_DELAY           | Arrival delay in minutes                                                     | "-19" (19 minutes early)                   |
| CRS_ELAPSED_TIME    | Scheduled flight duration in minutes                                         | "136"                                      |
| ACTUAL_ELAPSED_TIME | Actual flight duration in minutes                                            | "122"                                      |
| AIR_TIME            | Actual air time in minutes                                                   | "84"                                       |
| FLIGHTS             | Number of flights for the carrier                                            | "1"                                        |
| MONTH               | Month of the flight                                                           | "1" (January)                              |
| DAY_OF_MONTH        | Day of the month when the flight took place                                 | "1"                                        |
| DAY_OF_WEEK         | Day of the week the flight took place                                        | "1" (Monday)                               |
| ORIGIN_INDEX        | Index representing the origin airport's location                             | "166"                                      |
| DEST_INDEX          | Index representing the destination airport's location                        | "93"                                       |
| O_TEMP              | Temperature at the origin airport in Celsius                                 | "4.4"                                      |
| O_PRCP              | Precipitation at the origin airport in inches                                | "0"                                        |
| O_WSPD              | Wind speed at the origin airport in mph                                      | "11.2"                                     |
| D_TEMP              | Temperature at the destination airport in Celsius                            | "0"                                        |
| D_PRCP              | Precipitation at the destination airport in inches                           | "0"                                        |
| D_WSPD              | Wind speed at the destination airport in mph                                 | "13"                                       |
| O_LATITUDE          | Latitude of the origin airport                                               | "40.63975"                                 |
| O_LONGITUDE         | Longitude of the origin airport                                              | "-73.77893"                                |
| D_LATITUDE          | Latitude of the destination airport                                          | "42.21206"                                 |
| D_LONGITUDE         | Longitude of the destination airport                                         | "-83.34884"                                |
