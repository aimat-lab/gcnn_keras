.. _usage:
   :maxdepth: 3

Usage
=====

The basic idea of the interface is as follows: Create or load the physical main directory. Be careful and check where you place the main directory before generating a large number of jobs::


	from mjdir.MultiJobDirectory import MultiJobDirectory
	maindir = MultiJobDirectory("Name","filepath")

Then create multiple "jobs", for which empty physical subdirectories are automatically created by the python class::

	aindir.add("Calc_1")
	maindir.add(["Calc_2","Calc_3"])
	maindir.add({"Calc_4": {"command": 'echo "{path}" '} })

Get the current path list and information of all available directories via `get()` or for a specific sublist::

	maindir.get()  # list all
	maindir.get("Calc_1")

The class python dict holds a job plus path and additional information. You can delete entries via `remove()`. However, their physical subdirectories are not deleted!!::

	maindir.remove()  # remove all
	maindir.remove("Calc_1")

You can save and reload the python dict and also add existing directories that may not be in the pyhton dict if necessary::


	maindir.save() 
	maindir.load()
	maindir.load(add_existing=True)  # Can add all physical subdirectories without information

Create Input via own custom functions using libraries like ase or pymatgen that take a directory filepath as input.
The path can be obtained by `get()`. Some functions are found in [commands](mjdir/commands)::


	def write_input(filepath ):
	    # Do something
	    return 0	
	write_input( maindir.get()["Calc_1"]['path'] )

Modify and adjust queue settings:: 

	slurm_params = { 'tasks' : "10",
	                # "ntasks-per-node" : "10",
	                'time' : "30:00:00",
	                'nodes' : "1"}
	submit_properties = {'-p':'normal'}


And then run all jobs or a specific selection of available jobs with ``run()``. Here you can specify a number of properties. Like default command as string, the number of submits the jobs are distributed on and how many commands should be started asynchronously within one submission. The asynchronous execution must be compatible with the program and the system to use on. For more information see [commands](mjdir/commands) and [queue](mjdir/queue). The command is a string representing a bash command which is formatted by arguments provided by `add()` and enabled by `command_arguments`. Path information is available by default. Finally a set of bash scripts are generated and submitted. To inspect the submission without running, use `prepare_only=True` and look into the main directory::


	maindir.run(procs = 1,
		    asyn = 0,
 	            header="module purge",
    	            command= "# If not sepcified in add() use this",
            	    command_arguments = ['path'],
            	    queue_properties = slurm_params,
            	    submit_properties = submit_properties ,
            	    prepare_only = False)
