import numpy as np
import pandas as pd
from ColabDesign.Scfv import Scfv

class Eval:
    """ Class to evaluate designed ScFv sequences against target sequences
    """
    def __init__(self, mpnn_csv_path: str, ref_anti_fasta_path: str, ref_scfv_fasta_path: str, orientation_dict: dict, scheme = 'martin', linker = 'GGGGSGGGGSGGGGS', ):
        """ Initialization function for Eval class
        Args:
            mpnn_csv_path (str): Path to CSV file containing designed scFv sequences from MPNN
            og_anti_fasta_path (str): Path to FASTA file containing original antibody sequences
            ref_scfv_fasta_path (str): Path to FASTA file containing reference scFv sequences
            scheme (str): Numbering scheme to use for scFv or Paired (Heavy + Light) annotation (default: 'martin')
        """         
        self.mpnn_csv_path = mpnn_csv_path
        self.ref_anti_fasta_path = ref_anti_fasta_path
        self.ref_scfv_fasta_path = ref_scfv_fasta_path
        self.ref_paired_seq_annotator = Scfv(scheme=scheme)
        self.design_scfv_seq_annotator = Scfv(scheme=scheme)
        self.linker = linker
        self.orientation_dict = orientation_dict
    
    def load_refs(self):
        """ Function to load reference scFv sequences from FASTA file
        Returns:
            dict: Dictionary with keys being scFv IDs and values being sequences
        """ 
        # Load reference scFv sequences        
        ref_paired_seq_dict, ref_paired_seq_ids = self.ref_paired_seq_annotator.extract_seqs(self.ref_scfv_fasta_path, suffix_split = "_-_", seq_type = "", anti = False)

        # Load original antibody sequences
        ref_paired_seq_dict, ref_paired_seq_ids = self.ref_paired_seq_annotator.extract_seqs(self.ref_anti_fasta_path, suffix_split = "", seq_type = "", anti = True)

        self.ref_seq_dict = ref_paired_seq_dict
        self.ref_seq_ids = ref_paired_seq_ids

        # Annotate reference sequences
        self.annotated_ref_seqs = self.ref_paired_seq_annotator.annotate_seqs(self.linker, self.orientation_dict, generate_motif_commands = False)
        return self.annotated_ref_seqs
    
    def load_designs(self, scfv_ref_name: str):
        """ Function to load designed scFv sequences from MPNN CSV file
        Returns:
            dict: Dictionary with keys being scFv IDs and values being sequences
        """ 
        # Extract orientation, location of heavy FRs & CDRs, and location of light FRs & CDRs from reference name
        for ref_name, orientation in self.orientation_dict.items():
            if scfv_ref_name in ref_name:
                self.design_orientation = orientation
                self.design_heavy_region_loc_dict = self.annotated_ref_seqs[ref_name]['heavy']['region_loc_dict']
                self.design_light_region_loc_dict = self.annotated_ref_seqs[ref_name]['light']['region_loc_dict']
                break
        
        # Depending on design orientation, set how columns are named
        if self.design_orientation == 'VH-VL':
            col_names = {0 : 'heavy_seq', 1: 'light_seq'}
        elif self.design_orientation == 'VL-VH': # 'VL-VH'
            col_names = {0 : 'light_seq', 1: 'heavy_seq'}

        # Load designed scFv sequences from MPNN CSV
        df_designs = pd.read_csv(self.mpnn_csv_path, index_col=0)
        df_designs['scfv_seq'] = df_designs['seq'].str.split('/').str[0]
        df_seqs = df_designs['scfv_seq'].str.split(self.linker, expand=True).rename(columns=col_names)
        df_designs = pd.concat([df_designs, df_seqs], axis=1)
        df_designs_dict = df_designs[['design', 'n','scfv_seq', 'heavy_seq', 'light_seq']].to_dict(orient='records')

        
        
        # Create design scFv dictionary. Need to do this manually as AntPack does not correctly annotate designed sequences. Hints that designed seqs may not be valid and are not antibody-like
        annotated_design_seqs = {}
        for record in df_designs_dict:
            # Extract scFv ID, heavy and light sequences from record
            scfv_id = f"RF_design{record['design']}_num{record['n']}"
            heavy_seq = record['heavy_seq']
            light_seq = record['light_seq']

            # Initialize design scFv entry
            annotated_design_seqs[scfv_id] = {'heavy' : {}, 'light' : {}, 'seq' : '', 'orientation' : '', 'linker' : ''}
            
            # Annotate designed scFv sequences chain independent properties
            annotated_design_seqs[scfv_id]['orientation'] = self.design_orientation
            annotated_design_seqs[scfv_id]['linker'] = self.linker

            # Annotate designed scFv sequences chain dependent properties: region locations, region sequences, chain seq
            annotated_design_seqs[scfv_id]['heavy']['region_loc_dict'] = self.design_heavy_region_loc_dict
            annotated_design_seqs[scfv_id]['light']['region_loc_dict'] = self.design_light_region_loc_dict
            annotated_design_seqs[scfv_id]['heavy']['chain_seq'] = heavy_seq
            annotated_design_seqs[scfv_id]['light']['chain_seq'] = light_seq
            annotated_design_seqs[scfv_id]['heavy']['region_seqs_dict'] = {index : heavy_seq[locs['start']:locs['end']+1] for index, locs in self.design_heavy_region_loc_dict.items()}
            annotated_design_seqs[scfv_id]['light']['region_seqs_dict'] = {index : light_seq[locs['start']:locs['end']+1] for index, locs in self.design_light_region_loc_dict.items()}
            
            # Add full scFv sequence
            annotated_design_seqs[scfv_id]['seq'] = record['scfv_seq']
        
        self.annotated_design_seqs = annotated_design_seqs
        return self.annotated_design_seqs
    





    
