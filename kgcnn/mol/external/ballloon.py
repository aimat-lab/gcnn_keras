import sys
import os
import subprocess


class BalloonInterface:
    """Interface to Balloon program. This is tested for Balloon version 1.8.1.
    Described in http://dx.doi.org/10.1021/ci6005646
    Copyright (C) 2006-2021 Mikko J. Vainio and J. Santeri Puranen.
    Copyright (C) 2010 Visipoint Ltd. www.visipoint.fi.

    Setup Balloon configuration. This is the help parameter option of balloon help output mapped to python.
    The parameters are named identical except of `input_file`, `output_file` and `output_format`.

    Args:
        config (str): Name of a configuration file to read. Command-line options override the ones in the
            config file. A config file may contain lines that look like 'long_option_name = value'
            and comment lines that begin with '#'. Default is None.
        writeMOL2 (bool): Force output of structures in MOL2 format (can store partial atomic charges). See also
            'output-format'. Default is False.
        onlycharge (bool): Only assign partial atomic charges to the input structures and output in MOL2 format. A
            shortcut option for -c0 -k --writeMOL2. See also 'chargemodel'. Default is False.
        nobadmodels (bool): Do not write bad models to '<output-file>_bad<.suffix>'. Default is False.
        strict (bool): Skip structures that cannot be handled by the used force field. Default is False.
        nconfs (int): Number of conformers to generate or the initial population size if using GA. Zero
            will cause the input structure to be written out with partial atomic charges and energy as
            calculated by the MMFF94-like force field. Default is 1.
        randomSeed (int): Seed the pseudo-random number generator. Range [1, 4294967295). Default value taken
            from clock. Default is None.
        sdfnamefieldheader (str): When input is in MDL SD file format, use the contents of the data field with the
            given header as the molecule name. Only the first line of the data is considered.
        output_file (str): Name for the output file. If not given, the last filename in the input list is taken as
            the output file name. The file format is deduced from the file extension if not explicitly forced using
            'output-format'. Recognized extensions are sdf, sdf:v3, mol2, smi, vbf, xyz.
        output_format (str): Force output in the given format. Format is one of
            - sdf: Accelrys mol/sdf format.
            - sdf(v3): Accelrys mol/sdf format (V3000)
            - mol2: Tripos Mol2 format
            - smi: SMILES format
            - vbf: Visipoint Binary Format (experimental)
            - xyz: XMol XYZ format (experimental code)
        input_file (str): Input file name. File lists from multiple occurrences of the option are concatenated.
            If no output file is provided, the last name in the input file list is used as the output file name.
            Input file format is recognized automatically from the file contents. Recognized formats are
            - sdf: Accelrys mol/sdf format.
            - sdf(v3): Accelrys mol/sdf format (V3000).
            - mol2: Tripos Mol2 format.
            - smi: SMILES format. log: Gaussian98/03 output log format. (experimental)
            - pdb: Protein Data Bank format (experimental code)
            - vbf: Visipoint Binary Format (experimental)
            - xyz: XMol XYZ format (experimental code)
        testrun (bool): GA: Test the generated conformer ensemble against the input structure(s). The input
            geometries are used as reference for RMSD calculation, and ensemble statistics are calculated.
            Confomers are generated as if the input was 2D (connectivity only; see option 'rebuildGeometry').
            Typically, ligands from X-ray crystal structures are used as input for a testrun.
        noGA (bool): Do not use a genetic algorithm (GA) to generate conformations. Conformations will be
            generated via distance geometry.
        addConformerNumberToName (bool): Add the number of a conformer as a suffix to its name.
            The suffix is '_X', where X is the number. The name of the first conformer is not altered.
        listSymmetryClasses (bool): List the symmetry class identifiers for each atom of each input structure.
        singleconf (bool): Output only the lowest-energy conformation, regardless of the population size.
        fullforce (bool): GA: Use the full force field in the post-GA structure optimization. The default is to
            ignore torsion _gradient_ and all of electrostatics.
        nosymmetry (bool): GA: Disable the use of symmetry in calculation of RMSD. Symmetry perception can
            be very costly for large structures, e.g., proteins.
        useRingAtomsForRMSD (bool): Consider ring atoms for RMSD calculation instead of non-hydrogen atoms.
            Note that acyclic structures will not be processed at all.
        rebuildGeometry (bool): GA: Do not use the input geometry as template for conformer generation, but always
            rebuild the geometry. See also the 'testrun' option.
        maxPostprocessIter (int): GA: Maximum allowed number of iterations for conjugate gradient structure
            optimization in the post-processing phase. Default = 100.
        maxShapeIterations (int): GA: Maximum allowed number of iterations for overlap optimization in
            shape-matching. Default = 300. The truncated Newton optimizer is adapted from the TNPACK code of
            Schlick and co-workers: Schlick T and Fogelson A (1992) ACM TOMS 18, 46-70 & 71-111; Xie D and
            Schlick T (1999) SIAM J. Opt. 10, 132-154; Xie D and Schlick T (1999) ACM TOMS 25, 108-122.
        maxtime (int): GA: Maximum time [s] used per compound to evolve the GA. Default = 120.
        ftol (float): Tolerance for the change in the objective function value for terminating conjugate
            gradient structure optimization. Pass a negative value to omit this convergence
            criterion. Default is to omit.
        gradientTolerance (float): Tolerance for the gradient root-mean-square (RMS) value for terminating conjugate
            gradient structure optimization. Pass a negative value to omit this convergence criterion.
            Default = 0.050000000000000003.
        nicheRadius (foat): GA: Interconformer distance limit (RMSD) for the calculation of niche count.
            Default = 1.5
        RMSDtol (float): GA: Interconformer distance limit (RMSD) for the final pruning of conformers. If two
            conformers are closer than RMSDtol, the higher-energy conformer will be discarded. Default = 0.5
        energyConstant (int): GA: Constant term [kcal/mol] for the linear function of rotatable bonds used to
            calculate the potential energy window within which the final conformers must reside. Must be >= 0.
            Default = 10.
        energySlope (float): GA: Slope [kcal/mol/rotatable bond] for the linear function of rotatable bonds used to
            calculate the potential energy window within which the final conformers must reside. Must
            be >= 0. Default = 0.5.
        nGenerations (int): GA: The maximum number of generations. Default = 20.
        tournamentSize (int): GA: The number of individuals to involve in the tournament selection. The higher the
            number, the stronger the evolutionary pressure towards geometrically dissimilar results. Default = 2
        pTorsionMutation (float): GA: Mutation probability for changing a torsion angle value.
            Default = 0.050000000000000003
        pStereoMutation (float): GA: Mutation probability for inverting a stereochemical center.
            Default = 0.050000000000000003.
        pPyramidMutation (float): GA: Mutation probability for inverting a pyramidal center.
            Default = 0.050000000000000003
        pRingFlipMutation (float): GA: Mutation probability for a ring flip. Default = 0.050000000000000003
        parallelityThreshold (int): GA: Parallelity threshold [deg] for a ring flip to take effect.
            Valid values within the half-open interval [0,180). Default = 30
        pUniXO (float): GA: Uniform cross-over: probability for crossing. Tested at each locus.
            Default = 0.20000000000000001
        allowcrowded (bool): GA: Do not remove crowded solutions from the population during the evolutionary cycle.
        noPopulationGrowth (bool): GA: Do not allow the population to grow in order to accommodate the Pareto front.
        evolveOverlap (bool):  GA: Evolve the shape-density overlap with a template structure together with the
            conformational energy. Requires a template structure to be specified.
        adjusthydrogens (bool): Add hydrogens to the structures according to the octet rule.
            Hydrogens are always added to structures parsed from SMILES.
        neutralize (bool): Try to add or remove hydrogen atoms in order to make the input molecule neutral.
        stripSalts (bool): Remove all but the largest (highest number of atoms) connected component of the input
            structure.
        expand (bool): Distance geometry: Create expanded conformers. On each iteration, update the
            lower bounds of nonbonded atoms from the previously generated conformer.
        contract (bool): Distance geometry: Create contracted conformers. On each iteration, update the
            upper bounds of nonbonded atoms.
        stereo (bool): Always sample stereoconfigurations. Ignores any stereochemistry specifications in the input.
        keepInitial (bool): Output the initial geometry into the generated set of conformers.
        maxiter (int): Maximum allowed number of iterations for conjugate gradient structure optimization in
            the template geometry generation. Default = 100.
        useSimplex (bool): Use simplex downhill structure optimization prior to conjugate gradient.
        maxSimplexIterations (int): Maximum allowed number of iterations for downhill simplex structure
            optimization. Default = 500.
        simplexStepLength (int): The simplex edge length for downhill simplex structure optimization. Default = 1
        simplexFunctionTolerance (int): Tolerance for the relative change in the objective function value for
            terminating downhill simplex structure optimization. Default = 0.10000000000000001
        nInitialDimensions (int): The number of dimension in which to perform the initial geometry optimization
            after distance geometry. Increase if you encounter incorrect stereochemistry. Default = 4.
        forcefield (str): A filename to read for MMFF94 force field parameters. Alternatively, you can set
            environment variable named BALLOON_FORCEFIELD to point to the parameters file. The command
            line option overrides the environment variable.
        noVdWcutoff (bool): Do not use a cutoff distance for van der Waals (steric) energy evaluation.
        vdWCutoffOn (float): Distance at which the smoothing function for van der Waals energy cutoff is turned on.
        vdWCutoffOff (float): Distance at which the smoothing function for van der Waals energy cutoff is
            turned off.
        noEcutoff (bool): Do not use a cutoff distance for electrostatic energy evaluation.
        ECutoffOn (float): Distance at which the smoothing function for electrostatic energy cutoff is turned on.
        ECutoffOff (float): Distance at which the smoothing function for electrostatic energy cutoff is turned off.
        chargemodel (str): Specify the partial atomic charge model to be used.
            Alternatives are:
            - EEM: Puranen et al. (2010) J. Comput. Chem. 31, 1722-1732. doi:10.1002/jcc.21460
            - MMFF94: Halgren TA (1996) J. Comput. Chem. 17, 490-519.
            - SFKEEM: Puranen et al. (2010) J. Comput. Chem. 31, 1722-1732. doi:10.1002/jcc.21460
        distanceDependent (bool): Use a distance dependent dielectric model in optimizing the initial geometry.
        dielectric (float): Value of the dielectric constant aka relative static permittivity used in the
            calculation of electrostatic energy according to Coulomb's equation. Value must be > 1e-6 in
            order to avoid division by zero. For water, the value is 80 at room temperature. Defaults to one
            for vacuum.
        listAtomTypes (bool): Output a list of assigned force field atom types per each atom.
        maxRingSize (int): Maximum size for rings (#atoms) whose flexibility is to be handled. Negative values
            impose no limit (default).
        maxFlipDistance (int): A maximum allowed number of bonds between the pairs of ring atoms that define a
            flip-of-fragment operation. Negative values impose no limit. Default = 20
        query (str): GA: File name for a query structure upon which the compounds are overlaid based on
            shape-density overlap.
        noElitism (bool): Do not guarantee that the minimum energy conformer always survives to the next generation.
    """

    def __init__(self,
                 balloon_executable_path="",
                 config: str = None,
                 writeMOL2: bool = None,
                 onlycharge: bool = None,
                 nobadmodels: bool = None,
                 strict: bool = None,
                 nconfs: int = None,
                 randomSeed: int = None,
                 sdfnamefieldheader: str = None,
                 output_file: str = None,
                 output_format: str = None,
                 input_file: str = None,
                 testrun: bool = None,
                 noGA: bool = None,
                 addConformerNumberToName: bool = None,
                 listSymmetryClasses: bool = None,
                 singleconf: bool = None,
                 fullforce: bool = None,
                 nosymmetry: bool = None,
                 useRingAtomsForRMSD: bool = None,
                 rebuildGeometry: bool = None,
                 maxPostprocessIter: int = None,
                 maxShapeIterations: int = None,
                 maxtime: int = None,
                 ftol: float = None,
                 gradientTolerance: float = None,
                 nicheRadius: float = None,
                 RMSDtol: float = None,
                 energyConstant: int = None,
                 energySlope: float = None,
                 nGenerations: int = None,
                 tournamentSize: int = None,
                 pTorsionMutation: float = None,
                 pStereoMutation: float = None,
                 pPyramidMutation: float = None,
                 pRingFlipMutation: float = None,
                 parallelityThreshold: int = None,
                 pUniXO: float = None,
                 allowcrowded: bool = None,
                 noPopulationGrowth: bool = None,
                 evolveOverlap: bool = None,
                 adjusthydrogens: bool = None,
                 neutralize: bool = None,
                 stripSalts: bool = None,
                 expand: bool = None,
                 contract: bool = None,
                 stereo: bool = None,
                 keepInitial: bool = None,
                 maxiter: int = None,
                 useSimplex: bool = None,
                 maxSimplexIterations: int = None,
                 simplexStepLength: int = None,
                 simplexFunctionTolerance: int = None,
                 nInitialDimensions: int = None,
                 forcefield: str = None,
                 noVdWcutoff: bool = None,
                 vdWCutoffOn: float = None,
                 vdWCutoffOff: float = None,
                 noEcutoff: bool = None,
                 ECutoffOn: float = None,
                 ECutoffOff: float = None,
                 chargemodel: str = None,
                 distanceDependent: bool = None,
                 dielectric: float = None,
                 listAtomTypes: bool = None,
                 maxRingSize: int = None,
                 maxFlipDistance: int = None,
                 query: str = None,
                 noElitism: bool = None
                 ):
        """Initialize. Pass to config dictionary."""
        self.balloon_executable_path = balloon_executable_path
        self.input_file = input_file
        if self.input_file is not None:
            print("Definition of input file in config is not used. Please pass to run method.")
        self.output_format = output_format
        self.output_file = output_file

        self._config_args = {
            "config": config, "nconfs": nconfs, "randomSeed": randomSeed,
            "sdfnamefieldheader": sdfnamefieldheader, "maxPostprocessIter": maxPostprocessIter,
            "maxShapeIterations": maxShapeIterations, "maxtime": maxtime, "ftol": ftol,
            "gradientTolerance": gradientTolerance, "nicheRadius": nicheRadius, "RMSDtol": RMSDtol,
            "energyConstant": energyConstant, "energySlope": energySlope, "nGenerations": nGenerations,
            "tournamentSize": tournamentSize, "pTorsionMutation": pTorsionMutation, "pStereoMutation": pStereoMutation,
            "pPyramidMutation": pPyramidMutation, "pRingFlipMutation": pRingFlipMutation,
            "parallelityThreshold": parallelityThreshold, "pUniXO": pUniXO, "maxiter": maxiter,
            "maxSimplexIterations": maxSimplexIterations, "simplexStepLength": simplexStepLength,
            "simplexFunctionTolerance": simplexFunctionTolerance, "nInitialDimensions": nInitialDimensions,
            "forcefield": forcefield, "vdWCutoffOn": vdWCutoffOn, "vdWCutoffOff": vdWCutoffOff,
            "ECutoffOn": ECutoffOn, "ECutoffOff": ECutoffOff, "chargemodel": chargemodel, "dielectric": dielectric,
            "maxRingSize": maxRingSize, "maxFlipDistance": maxFlipDistance, "query": query
        }
        self._config_flags = {
            "writeMOL2": writeMOL2, "onlycharge": onlycharge, "nobadmodels": nobadmodels, "strict": strict,
            "testrun": testrun, "noGA": noGA, "addConformerNumberToName": addConformerNumberToName,
            "listSymmetryClasses": listSymmetryClasses, "singleconf": singleconf, "fullforce": fullforce,
            "nosymmetry": nosymmetry, "useRingAtomsForRMSD": useRingAtomsForRMSD, "rebuildGeometry": rebuildGeometry,
            "allowcrowded": allowcrowded, "noPopulationGrowth": noPopulationGrowth, "evolveOverlap": evolveOverlap,
            "adjusthydrogens": adjusthydrogens, "neutralize": neutralize, "stripSalts": stripSalts, "expand": expand,
            "contract": contract, "stereo": stereo, "keepInitial": keepInitial, "useSimplex": useSimplex,
            "noVdWcutoff": noVdWcutoff, "noEcutoff": noEcutoff, "distanceDependent": distanceDependent,
            "listAtomTypes": listAtomTypes, "noElitism": noElitism
        }

    def run(self, input_file: str, output_file: str = None, output_format: str = None):
        if input_file is None:
            raise ValueError("Input file must be defined.")
        if output_file is None:
            output_file = self.output_file
        if output_format is None:
            output_format = self.output_format

        command_list = [str(os.path.join(self.balloon_executable_path, "balloon"))]
        # Added flags. They have the same name as command line argument.
        for key, value in self._config_flags.items():
            if value is not None:
                command_list.append("--"+key)
        # Add args.
        for key, value in self._config_flags.items():
            if value is not None:
                command_list.append("--"+key)
                command_list.append(str(value))
        # Output format
        if output_format is not None:
            command_list.append("--output-format")
            command_list.append(str(output_format))

        # Finally add input and output file
        command_list.append("--input-file")
        command_list.append(str(input_file))

        if output_file is not None:
            command_list.append("--output-file")
            command_list.append(str(output_file))

        return_code = subprocess.run(command_list)
        # Check return code
        if int(return_code.returncode) != 0:
            raise ValueError("Batch process returned with error:", return_code)

        return return_code

    def get_config(self):
        config = {"input_file": self.input_file, "output_format": self.output_format, "output_file": self.output_file,
                  "balloon_executable_path": self.balloon_executable_path}
        config.update(self._config_args)
        config.update(self._config_flags)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
