# Data Generation

This directory contains the code used to construct the PathEval dataset.

## Environments

The environments are stored under ``./data`` subdirectory. Each ``.txt`` file where the first line is the number of polygons (obstacles) in the envrionment and the second line lists all of the polygons, but first specifying the number of points in the polygon followed by the coordinates of each point. 

## Path Generation

You can run the script ``generate_paths.sh`` to generate solutions for a subset of the environments under ``./data``. This allows for batching (i.e. generating solutions for multiple segments of the dataset in parallel). The script can be run as follows to generate paths for subset ``[start_id:end_id]`` of the environments

``./generate_paths.sh $n_samples $start_id $end_id $algorithm`` 

where,

- ``n_samples``: the number of solutions to propose for each environment.
- ``start_id``: the id of the first environment
- ``end_id``: the id of the last environment
- ``algorithm``: algorithm to be used to generate the solutions. While the paper only uses ``RRT``, the code also supports ``PRM`` (Any algorithm from the ``ompl`` library can be added seamlessly by modifying the ``GeneratePaths.cpp``)

## Pair Construction

Upon generating the paths, you can construct the pairs by running the script 

``./generate_samples.sh $paths-dir``

where, 

- ``paths_dir``: the directory where the paths generated in the step above are stored.