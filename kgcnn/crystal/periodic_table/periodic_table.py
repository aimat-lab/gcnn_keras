import pandas as pd
from typing import Optional
import importlib.resources
import kgcnn.crystal.periodic_table as periodic_table_module

# CSV file is downloaded from:
# https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV/?response_type=save&response_basename=PubChemElements_all

try:
    # >= Python 3.9
    periodic_table_csv = importlib.resources.files(periodic_table_module) / 'periodic_table.csv'
except:
    # < Python 3.9
    with importlib.resources.path(periodic_table_module, 'periodic_table.csv') as file_name:
        periodic_table_csv = file_name


class PeriodicTable:
    """Utility class to provide further element type information for crystal graph node embeddings."""
    
    def __init__(self, csv_path=periodic_table_csv,
                 normalize_atomic_mass=True,
                 normalize_atomic_radius=True,
                 normalize_electronegativity=True,
                 normalize_ionization_energy=True,
                 imputation_atomic_radius=209.46,  # mean value
                 imputation_electronegativity=1.18,  # educated guess (based on neighbour elements)
                 imputation_ionization_energy=8.):  # mean value
        self.data = pd.read_csv(csv_path)
        self.data['AtomicRadius'].fillna(imputation_atomic_radius, inplace=True)
        # Pm, Eu, Tb, Yb are inside the mp_e_form dataset, but have no electronegativity value
        self.data['Electronegativity'].fillna(imputation_electronegativity, inplace=True)
        self.data['IonizationEnergy'].fillna(imputation_ionization_energy, inplace=True)
        if normalize_atomic_mass:
            self._normalize_column('AtomicMass')
        if normalize_atomic_radius:
            self._normalize_column('AtomicRadius')
        if normalize_electronegativity:
            self._normalize_column('Electronegativity')
        if normalize_ionization_energy:
            self._normalize_column('IonizationEnergy')
            
    def _normalize_column(self, column):
        self.data[column] = (self.data[column] - self.data[column].mean()) / self.data[column].std()

    def get_symbol(self, z: Optional[int] = None):
        if z is None:
            return self.data['Symbol'].to_list()
        else:
            return self.data.loc[z-1]['Symbol']
    
    def get_atomic_mass(self, z: Optional[int] = None):
        if z is None:
            return self.data['AtomicMass'].to_list()
        else:
            return self.data.loc[z-1]['AtomicMass']
    
    def get_atomic_radius(self, z: Optional[int] = None):
        if z is None:
            return self.data['AtomicRadius'].to_list()
        else:
            return self.data.loc[z-1]['AtomicRadius']
    
    def get_electronegativity(self, z: Optional[int] = None):
        if z is None:
            return self.data['Electronegativity'].to_list()
        else:
            return self.data.loc[z-1]['Electronegativity']
    
    def get_ionization_energy(self, z: Optional[int] = None):
        if z is None:
            return self.data['IonizationEnergy'].to_list()
        else:
            return self.data.loc[z-1]['IonizationEnergy']

    def get_oxidation_states(self, z: Optional[int] = None):
        if z is None:
            return list(map(self.parse_oxidation_state_string, self.data['OxidationStates'].to_list()))
        else:
            oxidation_states = self.data.loc[z-1]['OxidationStates']
            return self.parse_oxidation_state_string(oxidation_states, encode=True)
    
    @staticmethod
    def parse_oxidation_state_string(s: str, encode: bool = True):
        if encode:
            oxidation_states = [0] * 14
            if isinstance(s, float):
                return oxidation_states
            for i in s.split(','):
                oxidation_states[int(i)-7] = 1
        else:
            oxidation_states = []
            if isinstance(s, float):
                return oxidation_states
            for i in s.split(','):
                oxidation_states.append(int(i))
        return oxidation_states
