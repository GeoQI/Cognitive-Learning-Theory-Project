# Cognitive-Learning-Theory-Project
Project for CS-541 Artificial Intelligence class


## Install
Clone the repository. You'll need numpy.


## Get Data
| Dataset         | URL                                                    |
|-----------------|--------------------------------------------------------|
| Human Resources | https://www.kaggle.com/ludobenistant/hr-analytics/data |

```sh
unzip human-resources-data.zip -d testfiles/
./bin/csv2tsv < testfiles/HR_comma_sep.csv > testfiles/HR_tab_sep.tsv

# Manually fix headers in the tsv to conform to pebl's data format (see below
# for link) and save in a different file (HR_pebl.tsv).

python . testfiles/HR_pebl.tsv
```
[Pebl's data format](https://pythonhosted.org/pebl/tutorial.html#pebl-s-data-file-format)


## Drawing the Graph
If you have `dot` installed, you can run:
```sh
python . testfiles/HR_pebl.tsv | dot -Tpng -o HR_network.png
```
And a png image of the graph will be generated.

Otherwise, the graphviz description for the network is outputted and can be
directed to a file and used in a standalone graphviz application, or
copy-pasted into a graphviz editor in order to generate a graph.

