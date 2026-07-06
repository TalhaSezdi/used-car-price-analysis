# Phase 1 Audit — Raw Dataset

**Rows:** 426,880  
**Columns:** 26  


## Missing Values

|              |   missing_count |   missing_pct |
|:-------------|----------------:|--------------:|
| county       |          426880 |        100    |
| size         |          306361 |         71.77 |
| cylinders    |          177678 |         41.62 |
| condition    |          174104 |         40.79 |
| VIN          |          161042 |         37.73 |
| drive        |          130567 |         30.59 |
| paint_color  |          130203 |         30.5  |
| type         |           92858 |         21.75 |
| manufacturer |           17646 |          4.13 |
| title_status |            8242 |          1.93 |
| lat          |            6549 |          1.53 |
| long         |            6549 |          1.53 |
| model        |            5277 |          1.24 |
| odometer     |            4400 |          1.03 |
| fuel         |            3013 |          0.71 |
| transmission |            2556 |          0.6  |
| year         |            1205 |          0.28 |
| description  |              70 |          0.02 |
| posting_date |              68 |          0.02 |
| image_url    |              68 |          0.02 |
| region_url   |               0 |          0    |
| url          |               0 |          0    |
| id           |               0 |          0    |
| region       |               0 |          0    |
| price        |               0 |          0    |
| state        |               0 |          0    |


## Numeric Distributions (price, odometer, year)

|       |            price |   odometer |         year |
|:------|-----------------:|-----------:|-------------:|
| count | 426880           | 422480     | 425675       |
| mean  |  75199           |  98043.3   |   2011.24    |
| std   |      1.21823e+07 | 213882     |      9.45212 |
| min   |      0           |      0     |   1900       |
| 1%    |      0           |      2     |   1967       |
| 5%    |      0           |   6318     |   1998       |
| 25%   |   5900           |  37704     |   2008       |
| 50%   |  13950           |  85548     |   2013       |
| 75%   |  26485.8         | 133542     |   2017       |
| 95%   |  44500           | 204000     |   2020       |
| 99%   |  66995           | 280000     |   2020       |
| max   |      3.73693e+09 |      1e+07 |   2022       |


## Categorical Value Counts (top 15 each)


### manufacturer

| manufacturer   |   count |
|:---------------|--------:|
| ford           |   70985 |
| chevrolet      |   55064 |
| toyota         |   34202 |
| honda          |   21269 |
| nissan         |   19067 |
| jeep           |   19014 |
| ram            |   18342 |
| nan            |   17646 |
| gmc            |   16785 |
| bmw            |   14699 |
| dodge          |   13707 |
| mercedes-benz  |   11817 |
| hyundai        |   10338 |
| subaru         |    9495 |
| volkswagen     |    9345 |



### condition

| condition   |   count |
|:------------|--------:|
| nan         |  174104 |
| good        |  121456 |
| excellent   |  101467 |
| like new    |   21178 |
| fair        |    6769 |
| new         |    1305 |
| salvage     |     601 |



### fuel

| fuel     |   count |
|:---------|--------:|
| gas      |  356209 |
| other    |   30728 |
| diesel   |   30062 |
| hybrid   |    5170 |
| nan      |    3013 |
| electric |    1698 |



### transmission

| transmission   |   count |
|:---------------|--------:|
| automatic      |  336524 |
| other          |   62682 |
| manual         |   25118 |
| nan            |    2556 |



### drive

| drive   |   count |
|:--------|--------:|
| 4wd     |  131904 |
| nan     |  130567 |
| fwd     |  105517 |
| rwd     |   58892 |



### type

| type        |   count |
|:------------|--------:|
| nan         |   92858 |
| sedan       |   87056 |
| SUV         |   77284 |
| pickup      |   43510 |
| truck       |   35279 |
| other       |   22110 |
| coupe       |   19204 |
| hatchback   |   16598 |
| wagon       |   10751 |
| van         |    8548 |
| convertible |    7731 |
| mini-van    |    4825 |
| offroad     |     609 |
| bus         |     517 |



### title_status

| title_status   |   count |
|:---------------|--------:|
| clean          |  405117 |
| nan            |    8242 |
| rebuilt        |    7219 |
| salvage        |    3868 |
| lien           |    1422 |
| missing        |     814 |
| parts only     |     198 |



### cylinders

| cylinders    |   count |
|:-------------|--------:|
| nan          |  177678 |
| 6 cylinders  |   94169 |
| 4 cylinders  |   77642 |
| 8 cylinders  |   72062 |
| 5 cylinders  |    1712 |
| 10 cylinders |    1455 |
| other        |    1298 |
| 3 cylinders  |     655 |
| 12 cylinders |     209 |



### paint_color

| paint_color   |   count |
|:--------------|--------:|
| nan           |  130203 |
| white         |   79285 |
| black         |   62861 |
| silver        |   42970 |
| blue          |   31223 |
| red           |   30473 |
| grey          |   24416 |
| green         |    7343 |
| custom        |    6700 |
| brown         |    6593 |
| yellow        |    2142 |
| orange        |    1984 |
| purple        |     687 |



### state

| state   |   count |
|:--------|--------:|
| ca      |   50614 |
| fl      |   28511 |
| tx      |   22945 |
| ny      |   19386 |
| oh      |   17696 |
| or      |   17104 |
| mi      |   16900 |
| nc      |   15277 |
| wa      |   13861 |
| pa      |   13753 |
| wi      |   11398 |
| co      |   11088 |
| tn      |   11066 |
| va      |   10732 |
| il      |   10387 |

