import os
import networkx as nx
from typing import Dict, Any, List, Set, Tuple

class EvidenceGraphManager:
    """
    Manages node operations, edge insertions, and query transactions.
    Supports Neo4j official connection, falling back automatically to an in-memory
    NetworkX multi-directed model if the server is offline or not configured.
    """
    def __init__(self):
        self.use_neo4j = False
        self.driver = None
        self.nx_graph = nx.MultiDiGraph()
        
        # Check environment configuration for Neo4j
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            from neo4j import GraphDatabase
            # Attempt connection
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.use_neo4j = True
            print("[Graph] Successfully connected to Neo4j instance at", neo4j_uri)
        except Exception as e:
            print(f"[Graph] Neo4j connection unavailable ({str(e)}). Falling back to local NetworkX engine.")
            self.use_neo4j = False

        # Load bootstrap sample data for demonstration of fraud rings
        self._bootstrap_sample_data()

    def add_node(self, node_id: str, node_type: str, attributes: Dict[str, Any]):
        attributes = attributes or {}
        attributes["node_type"] = node_type
        
        if self.use_neo4j:
            try:
                with self.driver.session() as session:
                    query = (
                        f"MERGE (n:{node_type} {{id: $id}}) "
                        f"SET n += $props"
                    )
                    session.run(query, id=node_id, props=attributes)
            except Exception as e:
                print(f"[Graph] Neo4j add_node error: {str(e)}")
        
        # Always maintain NetworkX graph copy for fallback and speed
        self.nx_graph.add_node(node_id, **attributes)

    def add_edge(self, source_id: str, target_id: str, edge_type: str, attributes: Dict[str, Any] = None):
        attributes = attributes or {}
        attributes["edge_type"] = edge_type
        
        if self.use_neo4j:
            try:
                # Find node types to ensure correct Cypher validation
                source_type = self.nx_graph.nodes[source_id].get("node_type", "User") if source_id in self.nx_graph else "User"
                target_type = self.nx_graph.nodes[target_id].get("node_type", "Device") if target_id in self.nx_graph else "Device"
                
                with self.driver.session() as session:
                    query = (
                        f"MATCH (a:{source_type} {{id: $source_id}}), (b:{target_type} {{id: $target_id}}) "
                        f"MERGE (a)-[r:{edge_type}]->(b) "
                        f"SET r += $props"
                    )
                    session.run(query, source_id=source_id, target_id=target_id, props=attributes)
            except Exception as e:
                print(f"[Graph] Neo4j add_edge error: {str(e)}")
                
        self.nx_graph.add_edge(source_id, target_id, key=edge_type, **attributes)

    def get_graph_data(self) -> Dict[str, Any]:
        """Returns JSON representation of nodes and edges for front-end D3/SVG viewer."""
        nodes = []
        for n_id, n_data in self.nx_graph.nodes(data=True):
            nodes.append({
                "id": n_id,
                "label": n_data.get("label", n_id),
                "type": n_data.get("node_type", "User"),
                "status": n_data.get("status", "VERIFIED"),
                "risk_score": n_data.get("risk_score", 0.1)
            })
            
        links = []
        for u, v, key, e_data in self.nx_graph.edges(data=True, keys=True):
            links.append({
                "source": u,
                "target": v,
                "type": key,
                "label": e_data.get("edge_type", key)
            })
            
        return {"nodes": nodes, "links": links}

    def detect_fraud_rings(self) -> List[List[str]]:
        """
        Clustering & fraud-ring detection algorithm.
        Finds sets of Users sharing identical Device IDs, IP coordinates, or Face embeddings.
        """
        # Convert multi-directed graph to undirected graph for connected component calculation
        undirected_g = self.nx_graph.to_undirected()
        components = list(nx.connected_components(undirected_g))
        
        fraud_rings = []
        for comp in components:
            # We classify as a fraud ring if a cluster contains multiple users sharing device/IP/face links
            users_in_comp = [node for node in comp if self.nx_graph.nodes[node].get("node_type") == "User"]
            if len(users_in_comp) >= 2:
                # Check if it has sharing connections
                sharing_edges = False
                for u in users_in_comp:
                    for v in users_in_comp:
                        if u != v:
                            # If they share an IP, device, face, address
                            paths = list(nx.all_simple_paths(undirected_g, u, v, cutoff=2))
                            if len(paths) > 0:
                                sharing_edges = True
                                break
                if sharing_edges:
                    fraud_rings.append(list(comp))
                    
        return fraud_rings

    def traverse_subgraph(self, start_node_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Traverses relationships to retrieve RAG context for a user validation file."""
        if start_node_id not in self.nx_graph:
            return []
            
        # Traverse undirected to find all close neighbors (devices, networks, profiles)
        undirected_g = self.nx_graph.to_undirected()
        subgraph_nodes = nx.single_source_shortest_path_length(undirected_g, start_node_id, cutoff=max_depth)
        
        logs = []
        for node_id in subgraph_nodes:
            # Describe nodes
            n_data = self.nx_graph.nodes[node_id]
            logs.append({
                "id": node_id,
                "type": n_data.get("node_type"),
                "attributes": {k: v for k, v in n_data.items() if k != "node_type"}
            })
            
        # Describe all actual directed edges (in and out) between nodes in the subgraph
        for u, v, key, e_data in self.nx_graph.edges(data=True, keys=True):
            if u in subgraph_nodes and v in subgraph_nodes:
                logs.append({
                    "relation": key,
                    "source": u,
                    "target": v,
                    "attributes": e_data
                })
        return logs

    def _bootstrap_sample_data(self):
        """Pre-populates demonstration data illustrating a sophisticated fraud ring."""
        # Fraud Ring 1: Shared Device & Mismatched Faces
        self.add_node("user_alice", "User", {"label": "Alice Smith", "status": "VERIFIED", "risk_score": 0.12})
        self.add_node("user_bob", "User", {"label": "Bob Johnson", "status": "PENDING", "risk_score": 0.65})
        self.add_node("user_charlie", "User", {"label": "Charlie Davis", "status": "PENDING", "risk_score": 0.88})
        
        self.add_node("device_mac_001", "Device", {"label": "MacBook Pro M1 (ID: 001)", "model": "Apple MacBook"})
        self.add_node("ip_192_168_1_50", "IP", {"label": "IP 192.168.1.50", "geo": "New York, USA"})
        self.add_node("face_profile_alpha", "Face", {"label": "Biometric Face Template Alpha"})
        
        self.add_edge("user_alice", "device_mac_001", "SAME_DEVICE")
        self.add_edge("user_bob", "device_mac_001", "SAME_DEVICE")
        self.add_edge("user_charlie", "device_mac_001", "SAME_DEVICE")
        
        self.add_edge("user_alice", "ip_192_168_1_50", "SAME_NETWORK")
        self.add_edge("user_bob", "ip_192_168_1_50", "SAME_NETWORK")
        
        self.add_edge("user_charlie", "face_profile_alpha", "SAME_FACE")
        self.add_edge("user_bob", "face_profile_alpha", "SAME_FACE") # Bob and Charlie share the same facial scan under different names!
        
        # Fraud Ring 2: Separate clean user
        self.add_node("user_david", "User", {"label": "David Wilson", "status": "VERIFIED", "risk_score": 0.08})
        self.add_node("device_iphone_007", "Device", {"label": "iPhone 15 (ID: 007)"})
        self.add_edge("user_david", "device_iphone_007", "SAME_DEVICE")
