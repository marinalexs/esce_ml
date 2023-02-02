# empirical sample complexity estimator

## example run

install environment defined in `environment.yml`

take a look at the files in the `example` folder...

then run the workflow:

`snakemake --configfile example/example_config.yaml --cores 1 --rerun-incomplete all`

view results

`streamlit run server.py`

if not running on a local machine you may need to forward the relevant ports, or compress and download results and run streamlit locally

`tar -cvpzf results.tar.gz results/example-dataset/statistics results/example-dataset/scores`

`streamlit run server.py`


