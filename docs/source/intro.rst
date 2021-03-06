.. _intro:
   :maxdepth: 3

Introduction
============

The concept of this pyhton class is to have a neat interface to manage a folder with multiple jobdirectories.
Having a managed directory, jobs can be submitted from via a queueing system like slurm and distributed in folders.
The goal is to be able to submit array-jobs via python, providing the same interface as bash and queueing like slurm.
In [commands](mjdir/commands), modules should be collected that are used to generate and read input for specific task and programs.
The main class is sought to have as little dependencies as possible, ideally none.
The directory management should be os-independent, the submission is not. For the moment only slurm is supported. 