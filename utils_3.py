from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Atom, BondType

import numpy as np
import pandas as pd

from rdkit.Chem import rdFMCS

from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def canonize_smiles(s):
    """ canonizes the given smiles """
    m = Chem.MolFromSmiles(s)
    Chem.Kekulize(m,clearAromaticFlags=True)
    s_can = Chem.MolToSmiles(m,isomericSmiles=True)
    
    return s_can


def amat_entry(amat,r,c,val):
    """
    changes the (r,c) and (c,r) values of matrix "amat" to "val"
    """
    amat[r-1][c-1] = amat[c-1][r-1]= val
    return


def amat_edit(amat,r,c,delta):
    """
    changes the (r,c) and (c,r) values of matrix "amat" to "val"
    """
    amat[r-1][c-1] += delta
    if r != c:
        amat[c-1][r-1] += delta
    return


def sum_abs_adj_matrix(m):
    return sum(sum(np.abs(m)))/2

def get_core_atom_count(mat,start_atom=0):
    """ obtain number of atoms in the final target. 
        usually assumes the 0-index atom is in the target.
        can be changed by setting another index as starting_atom
        this can also be used to get the size of any other fragment,
        given the user knows one of its atom indices """
    atoms_visited = []
    atoms_to_visit = [start_atom]

    while len(atoms_to_visit) > 0:
        
        atom_check = atoms_to_visit.pop(0)
        atoms_visited.append(atom_check)

        atoms_connected = np.where(mat[atom_check] > 0)[0]

        atoms_not_visited = [i for i in atoms_connected if i not in atoms_visited+atoms_to_visit]
        atoms_to_visit.extend(atoms_not_visited)

    return len(atoms_visited)




def molFromAdjMat(atoms, amat,sanitize=True):
    """Creates a mol object from an adjacency matrix.
    Inputs:
    atoms: list of atomic numbers of atoms, by row
    amat: adjacency matrix. Has to have same length as atoms (obviously)
    sanitize: bool, whether to RDKit-sanitize mol before return. 
        Used to check for errors.
    Output: mol object
    """
    
    m = Chem.RWMol()
    # add in the separate atoms
    for a in atoms: m.AddAtom(Atom(int(a)))
    side_len = len(amat)    
    for r in range(side_len):
        for c in range(r+1,side_len):
            bond_order = amat[r][c]
            if bond_order > 0:
                if bond_order == 1: m.AddBond(r,c,BondType.SINGLE)
                if bond_order == 2: m.AddBond(r,c,BondType.DOUBLE)
                if bond_order == 3: m.AddBond(r,c,BondType.TRIPLE)

    if sanitize:
        Chem.SanitizeMol(m)
    return m

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1)
    return mol


def make_changelogs(data_file_path):
    data = pd.read_csv(data_file_path)[["bond","edit","file"]]
    data = data[~data.bond.isnull()].copy()
    
    changelogs = []
    entry = {}
    bond_edits = ()
    for r in data.itertuples():


        # reset entry dict at new step
        if r[1] == "step":
            print(r[2])
            entry["edits"] = bond_edits
            changelogs.append(entry)
            entry = {}

        # if padding, fill th
        elif r[1] == "pad":
            pad_atoms = r[3].split(" ")
            pad_atoms = [int(i) for i in pad_atoms]

            if pad_atoms == [0]:
                entry["pad"] = 0
                entry["pad_elem"] = []

            else:
                entry["pad"] = len(pad_atoms)
                entry["pad_elem"] = pad_atoms

            # make the empty bond edits here to prepare
            bond_edits = []
        else:
            bond_edits.append((int(r[1]),int(r[2]),int(r[3])))
    
    return changelogs



def apply_changes(amat_init, atoms_init,changelogs):
    seq_out = [amat_init.copy()]
    amat = amat_init.copy()
    atoms = atoms_init.copy()
    for i in changelogs:
        try:
            
            pad_amt = i["pad"]

            if pad_amt > 0:
                amat = np.pad(amat,[(0, pad_amt), (0, pad_amt)],  mode="constant")
                atoms.extend(i["pad_elem"])

            for ed in i["edits"]:
                amat_edit(amat,ed[0],ed[1],ed[2])
            seq_out.append(amat.copy())
            
        except Exception as e:
            print(i)
            print(str(e))
            print(repr(e))
        
    seq_out.reverse()
    
    all_sizes = [m.shape[0] for m in seq_out]
    max_size = max(all_sizes)

    output_padded = []

    for mat in seq_out:
        mat_size = mat.shape[0]
        if mat_size < max_size:
            pad_size = max_size - mat_size 
            output_padded.append(np.pad(mat, [(0, pad_size), (0, pad_size)], mode='constant'))
        else:
            output_padded.append(mat)
            
        
    return output_padded,atoms


def get_common_atom_bond(mols,use_bond_order=True):
    common_atoms_num = []
    common_bonds_num = []
    for mol in mols:
        pair = [mol, mols[-1]]
        if use_bond_order:
            mcs = rdFMCS.FindMCS(pair,matchValences=True)
        else:
            mcs = rdFMCS.FindMCS(pair,bondCompare=rdFMCS.BondCompare.CompareAny)
        common_atoms_num.append(mcs.numAtoms)
        common_bonds_num.append(mcs.numBonds)
    return common_atoms_num,common_bonds_num


def get_mols_from_matrix(route,atoms,priority_atoms=np.array([0,0])):
    mols = []
    
    blank_row = np.array([0,0])
    priority_atoms = np.vstack((blank_row,priority_atoms))
    
    for i_a, amat in enumerate(route):
        m = molFromAdjMat(atoms,amat)
        
        if i_a+1 in priority_atoms[:,0]:
            pri_pair_loc = np.where(priority_atoms[:,0]==i_a+1)[0][0]

            pri_atom = priority_atoms[pri_pair_loc][1]
            
            pri_frag_tf = np.array([pri_atom in f for f in Chem.GetMolFrags(m)])
            pri_frag_ind = np.where(pri_frag_tf)[0][0]
            
            priority_mol = Chem.GetMolFrags(m,asMols=True)[pri_frag_ind]
            mols.append(priority_mol)
            
        else:

            frag_sizes = [len(i) for i in Chem.GetMolFrags(m)]
            max_frag_index = frag_sizes.index(max(frag_sizes))
            largest_mol = Chem.GetMolFrags(m,asMols=True)[max_frag_index]
            mols.append(largest_mol)
        
    return mols


def get_distances(r1,use_conns=False):
    
    total_diffs = []
    stereos = [np.diag(amat.copy()) for amat in r1]
    final_stereo = stereos[-1]
    
    # use unweighted bond matrix if needed
    if use_conns:
        r1 = (r1!=0).astype(int)
        
    # get a version with blank diagonals (since stereochem is processed separately)
    final_bonds = r1[-1].copy()
    np.fill_diagonal(final_bonds, 0)
    
    for i_rm, raw_mat in enumerate(r1):
        
        # calculate stereochem distance 
        stereo_data = stereos[i_rm]
        stereo_dist = sum(stereo_data != final_stereo)
        
        # get bonds only (without stereo)
        mat = raw_mat.copy()
        np.fill_diagonal(mat, 0)
        
        # bond distance
        diff = final_bonds - mat
        to_break = -np.sum(diff[diff<0])/2
        to_form  =  np.sum(diff[diff>0])/2
        
        # total edit distance
        total_diff = sum_abs_adj_matrix(diff) + stereo_dist
        total_diffs.append([to_break,to_form,stereo_dist,total_diff])
        
    total_diffs = np.array(total_diffs)
    step_diffs = []
    stereo_diffs = np.ediff1d(total_diffs[:,2])
    for i_mat in range(len(r1)-1):
        # difference between 2 intermediates
        step_diff = r1[i_mat+1] - r1[i_mat]
        
        # zero out the stereochem 
        np.fill_diagonal(step_diff, 0)
        
        # total bond and stereocenter edits between 2 consecutive intermediates 
        total_diff_abs = sum_abs_adj_matrix(step_diff) + abs(stereo_diffs[i_mat])

        step_diffs.append(total_diff_abs)
        
    return {"total_diffs":total_diffs, "step_diffs":step_diffs}


def get_route_stereo_score(route):
    
    scores = []
    for step_number in range(1,len(route)):
        
        make_correct_stereo, make_inverted_stereo, flip_stereo = 0,0,0
        diag_before = np.diag(route[step_number-1].copy())
        diag_after =  np.diag(route[step_number].copy())

        for i in zip(diag_before,diag_after):
            if i == (0,1):
                make_correct_stereo += 1
            if i == (0,-1):
                make_inverted_stereo += 1

            if i == (-1,1):
                flip_stereo += 1
                
        scores.append([make_correct_stereo, make_inverted_stereo, flip_stereo])
        
    return np.array(scores)
    
def get_route_score(route,target_atom_count,use_conns=False,as_df=False):
    
    
    stereo_score = get_route_stereo_score(route.copy())
    # use unweighted bond matrix if needed
    if use_conns:
        route = (route!=0).astype(int)
    
    amat_target = route[-1].copy()
    scores = []
    for step_number in range(1,len(route)):
        amat_before = route[step_number-1]
        amat_after = route[step_number]

        amat_change = amat_after - amat_before
        
        changing_bonds = np.where(amat_change !=0)
        anno_bonds = np.vstack(changing_bonds).T
        
        target_bonds = amat_target[changing_bonds]

        before_bonds = amat_before[changing_bonds]
        before_distance = target_bonds - before_bonds

        after_bonds = amat_after[changing_bonds]
        after_distance = target_bonds - after_bonds
        
        t_f_strat, t_f_con, t_b_strat,t_b_con,c_f_fin, c_f_con, c_b_con = 0,0,0,0,0,0,0

        for i_b, b in enumerate(anno_bonds):

            dupe_flag = False 
            form_strat_flag = False
            lower_tri = b[0] > b[1]
            is_stereocenter = b[0] == b[1]
            in_concession_region = any(b >= target_atom_count)
            in_target_region = not in_concession_region

            if lower_tri:
                if in_target_region:
                    # formation of strategic bond
                    if before_distance[i_b] > 0 and after_distance[i_b] < before_distance[i_b]:
                        t_f_strat += before_distance[i_b] - np.max([after_distance[i_b],0])
                        form_strat_flag = True
                        dupe_flag=True

                    # formation of concession bond
                    if after_distance[i_b] < 0 and after_distance[i_b] < before_distance[i_b]:

                        t_f_con += after_distance[i_b] - np.min([before_distance[i_b],0])
                        
                        if dupe_flag and not form_strat_flag:
                            print("dupe conditions")
                            print(f"{step_number}, {before_distance[i_b]}, {after_distance[i_b]}")
                        else: dupe_flag=True

                    # breaking strategic bond (never personally seen, included for completion)
                    if after_distance[i_b] > 0 and after_distance[i_b] > before_distance[i_b]:
                        t_b_strat += before_distance[i_b] - after_distance[i_b]

                        if dupe_flag:
                            print("dupe conditions")
                            print(f"{step_number}, {before_distance[i_b]}, {after_distance[i_b]}")
                        else: dupe_flag=True

                    # breaking concession bond
                    if before_distance[i_b] < 0 and after_distance[i_b] > before_distance[i_b]:
                        t_b_con += after_distance[i_b] - before_distance[i_b]

                        if dupe_flag:
                            print("dupe conditions")
                            print(f"{step_number}, {before_distance[i_b]}, {after_distance[i_b]}")
                        else: dupe_flag=True

                    if not dupe_flag:
                        print(f"something fell through (strat), {before_distance[i_b]}, {after_distance[i_b]}")

                elif in_concession_region:
                    
                    if before_distance[i_b] > 0 and after_distance[i_b] < before_distance[i_b]:
                        c_f_fin += before_distance[i_b] - after_distance[i_b]
                        dupe_flag=True

                    # formation of concession bond
                    if after_distance[i_b] < 0 and after_distance[i_b] < before_distance[i_b]:
                        c_f_con += after_distance[i_b] - before_distance[i_b]
                        if dupe_flag:
                            print("dupe conditions")
                            print(f"{step_number}, {before_distance[i_b]}, {after_distance[i_b]}")
                        else: dupe_flag=True

                    # breaking concession bond
                    if before_distance[i_b] < 0 and after_distance[i_b] > before_distance[i_b]:
                        c_b_con += after_distance[i_b] - before_distance[i_b]

                        if dupe_flag:
                            print("dupe conditions")
                            print(f"{step_number}, {before_distance[i_b]}, {after_distance[i_b]}")
                        else: dupe_flag=True

                    if not dupe_flag:
                        print(f"something fell through (conn), {step_number}, {before_distance[i_b]}, {after_distance[i_b]}")
                    
        scores.append([t_f_strat, t_f_con, t_b_strat,t_b_con,c_f_fin, c_f_con, c_b_con])
        
    
    if as_df:
#         scores_out = np.array(scores)
        step_numbers = np.array([np.arange(1,len(scores)+1,1)])
        array_out = np.hstack((step_numbers.T,np.array(scores),stereo_score))
        col_labels = ["step number","t_f_strat", "t_f_con", "t_b_strat","t_b_con","c_f_fin", "c_f_con", "c_b_con", 
                                    "make_correct_stereo", "make_inverted_stereo", "flip_stereo"]
        
        
        return pd.DataFrame(data=np.array(array_out),columns=col_labels)
    else:
        return np.hstack((np.array(scores),stereo_score))
    
    
def get_hybrid_score(route,target_atom_count):
#     "t_f_strat", "t_f_con", "t_b_strat","t_b_con","c_f_fin", "c_f_con", "c_b_con", 
#                                     "make_correct_stereo", "make_inverted_stereo", "flip_stereo"
    score_bond = get_route_score(route,target_atom_count,as_df=False,use_conns=False)
    score_conn = get_route_score(route,target_atom_count,as_df=False,use_conns=True)
    # concession, forming final
    score_bond[:,4] = 0
    
    # target, breaking concession
    score_bond[:,3] = score_conn[:,3]
    
    # concession, breaking concession
    score_bond[:,6] = score_conn[:,6] 
    
    return np.sum(score_bond,axis=1), score_bond

def get_os_scores(amats,target_atom_count,region="target"):

    # calculate "oxidation state" differences
    os_hist = get_connections(amats)
    target_final_os     = os_hist[-1][:target_atom_count]
    concession_final_os = os_hist[-1][target_atom_count:]

    target_os_diffs = []
    concession_os_diffs = []
    os_step_diffs = []
    
    for i_r, os_row in enumerate(os_hist):
        target_os     = os_row[:target_atom_count]
        concession_os = os_row[target_atom_count:]

        target_diff     = np.sum(np.abs(target_final_os-target_os))
        concession_diff = np.sum(np.abs(concession_final_os-concession_os))
        if i_r < len(os_hist)-1:
            before_os = os_row[:target_atom_count]
            after_os = os_hist[i_r+1][:target_atom_count]
            os_step_diffs.append(np.sum(np.abs(after_os-before_os)))

        target_os_diffs.append(target_diff)
        concession_os_diffs.append(concession_diff)

    target_os_diffs = np.array(target_os_diffs)    
    concession_os_diffs = np.array(concession_os_diffs)
    
    if region == "target":
        return target_os_diffs,os_step_diffs
    elif region == "concession":
        return concession_os_diffs,os_step_diffs
    
    else:
        print("region = target / concession")
        return

def get_connections(route):
    all_conns = []
    for amat in route:
        conns = []

        # backup
        temp_amat = amat.copy()
        np.fill_diagonal(temp_amat,0)

        for row in temp_amat:
            # total connectivity
            conns.append(np.sum(row))

        all_conns.append(conns)
        
    return np.array(all_conns)

dblue = [13,114,186,256]
lblue = [46,191,206,256]
black = [0,0,0,256]
lpink = [234,151,192,256]
mpink = [203,46,139,256]
dpink = [142,36,101,256]

synthia_cmap = np.array([dblue,lblue,black,lpink,mpink,dpink])/256
synthia_pm_cmap = np.array([dblue,black,mpink])/256