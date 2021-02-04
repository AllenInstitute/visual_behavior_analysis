# Overview of qc visualization

## To run the QC app:
at the command line, run:  

    python ~/visual_behavior/visualization/qc/dash_app/app.py

The app will then be visible at `{COMPUTER NAME}.corp.alleninstitute.org:{PORT NUMBER}`  
Optional command line arguments:  

    --port {PORT NUMBER} # default = 3389
    --debug # boolean, puts app in debug mode if flag is passed

Port argument: port 3389, the default port, is the remote desktop port used by Windows computers. The institute lets this port pass through VPN.  
Debug argument: if passed, puts the app into debug mode. This makes the app refresh whenever the underlying source code is updated. This is good to get quick feedback on changes (including knowing if syntax errors break the app). But it is undesirable if users are actively accessing the app.

## Folder structure for plots
container level plots should be saved in the following folder:  
`/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/container_plots`

Each distinct plot type should have a folder name, with each plot in the folder named as container_{CONTAINER_ID}.png

## Formatting the JSON file
Inside of the folder above, there needs to be a file called `qc_definitions.json` formatted as follows:  
'''
{
  "DISPLAY NAME 1": {
    "show_plots": true,
    "plot_folder_name": "folder_1",
    "qc_attributes": ["no_problems", "missing_visualization", "flag_as_problematic"]
  },
  "DISPLAY NAME 2": {
    "show_plots": false,
    "qc_attributes": ["no_problems", "missing_visualization", "flag_as_problematic", "another_tag"]
  },
}
'''
where
* DISPLAY NAME attributes control what is shown in the app
* show_plots controls whether or not the plots are displayed by the app
* plot_folder_name is the name of the folder where the plots are saved
* qc_attributes is the list of attributes that users should be able to choose from when qc'ing this plot

## Generating new plots
container level plotting functions are in `visual_behavior/visualization/qc/container_plots.py`
A function to call plotting code is in `visual_behavior/visualization/qc/save_all_container_plots.py`
A function to deploy plotting code to the cluster is in `visual_behavior/visualization/qc/run_save_all_container_plots.py`