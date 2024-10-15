# pdb_data_processing.py

import os
import re
import math
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

def dist(p1, p2):
    """计算两个三维点之间的欧几里得距离"""
    return math.sqrt((p1[0] - p2[0])**2 + 
                     (p1[1] - p2[1])**2 + 
                     (p1[2] - p2[2])**2)

def read_atoms(file, chain=".", model=1):
    """
    读取PDB文件，提取指定链中的CA原子的坐标和氨基酸类型
    :param file: PDB文件对象
    :param chain: 指定链的ID（默认所有链）
    :param model: 指定模型（未使用）
    :return: atoms (list of tuples), aa_types (list of str)
    """
    pattern = re.compile(chain)
    atoms = []
    aa_types = []
    for line in file:
        line = line.strip()
        if line.startswith("ATOM"):
            atom_type = line[12:16].strip()
            chain_id = line[21:22]
            if atom_type == "CA" and re.match(pattern, chain_id):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                aa_type = line[17:20].strip()
                atoms.append((x, y, z))
                aa_types.append(aa_type)
    return atoms, aa_types

def compute_contacts(atoms, threshold):
    """
    根据距离阈值计算氨基酸之间的接触对
    :param atoms: list of tuples
    :param threshold: 距离阈值
    :return: list of tuples (i, j)
    """
    contacts = []
    for i in range(len(atoms)-2):
        for j in range(i+2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i, j))  # 0-based indexing
    return contacts

def assign_features(aa_list, all_for_assign):
    """
    为每个氨基酸分配特征向量
    :param aa_list: list of str
    :param all_for_assign: numpy array, shape (20, feature_dim)
    :return: numpy array, shape (num_nodes, feature_dim)
    """
    aa_mapping = {
        'ALA':0, 'CYS':1, 'ASP':2, 'GLU':3, 'PHE':4, 'GLY':5,
        'HIS':6, 'ILE':7, 'LYS':8, 'LEU':9, 'MET':10, 'ASN':11,
        'PRO':12, 'GLN':13, 'ARG':14, 'SER':15, 'THR':16, 'VAL':17,
        'TRP':18, 'TYR':19
    }
    num_features = all_for_assign.shape[1]
    x_p = np.zeros((len(aa_list), num_features))
    for j, aa in enumerate(aa_list):
        if aa in aa_mapping:
            x_p[j] = all_for_assign[aa_mapping[aa], :]
        else:
            x_p[j] = np.zeros(num_features)  # 处理未知氨基酸类型
    return x_p

def build_graph(atoms, contacts):
    """
    根据接触对构建图的边索引
    :param atoms: list of tuples
    :param contacts: list of tuples (i, j)
    :return: torch.LongTensor, shape (2, num_edges)
    """
    edge_index = []
    for (i, j) in contacts:
        edge_index.append([i, j])
        edge_index.append([j, i])  # 无向图，添加双向边
    if not edge_index:
        return torch.empty((2,0), dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def load_protein_names(csv_path):
    """
    从CSV文件加载蛋白质名称
    :param csv_path: str
    :return: list of str
    """
    df = pd.read_csv(csv_path)
    protein_names = df.iloc[:,0].tolist()  # 假设蛋白质名称在第一列
    return protein_names

def load_all_assign(all_assign_path):
    """
    加载all_assign.txt文件
    :param all_assign_path: str
    :return: numpy array
    """
    return np.loadtxt(all_assign_path)

def process_proteins(csv_path, pdb_folder, all_assign_path, distance_threshold=7.5):
    """
    处理所有蛋白质，提取图数据
    :param csv_path: str
    :param pdb_folder: str
    :param all_assign_path: str
    :param distance_threshold: float
    :return: list of Data
    """
    protein_names = load_protein_names(csv_path)
    all_for_assign = load_all_assign(all_assign_path)
    data_list = []
    
    for protein in tqdm(protein_names, desc="Processing proteins"):
        pdb_file = os.path.join(pdb_folder, f"{protein}.pdb")
        if not os.path.exists(pdb_file):
            print(f"Warning: PDB file {pdb_file} does not exist.")
            continue
        with open(pdb_file, 'r') as f:
            atoms, aa_types = read_atoms(f)
        if not atoms:
            print(f"Warning: No CA atoms found in {pdb_file}.")
            continue
        contacts = compute_contacts(atoms, distance_threshold)
        edge_index = build_graph(atoms, contacts)
        if edge_index.size(1) == 0:
            print(f"Warning: No contacts found in {pdb_file}.")
            continue
        features = assign_features(aa_types, all_for_assign)
        data = Data(x=torch.tensor(features, dtype=torch.float),
                    edge_index=edge_index,
                    num_nodes=len(features))
        data_list.append(data)
    
    return data_list