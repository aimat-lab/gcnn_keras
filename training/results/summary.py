import os


benchmark_datasets = {
    "CoraDataset": {"general_info": "Cora Dataset  "}
}


with open("README.md", "w") as f:
    f.write("# Summary of Benchmark Training\n\n")

    for dataset, dataset_info in benchmark_datasets.items():
        f.write("## %s\n\n" % dataset)
        f.write("%s\n" % dataset_info["general_info"])

