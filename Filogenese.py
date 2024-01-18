import tensorflow as tf
import numpy as np
import ltn
import networkx as nx
import itertool


Ancestral = ltn.Predicate.MLP([embedding_size, embedding_size], hidden_layer_sizes = (8,8))
Descendente = ltn.Predicate.MLP([embedding_size, embedding_size], hidden_layer_sizes = (8,8))

Monera = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))
Protista =ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))

# Reino Animal 
# ******************************
# Cordados e Vertebrados
Mamifero = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes =(8,8)) 
Ave = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))
Reptil =ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))
Anfibio = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))
Peixe = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))

# Invertebrados
Equinodermo = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))
Platelminto = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))
Nematelminto = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))
Molusco = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))
Anelidos = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))
Artropodes = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))
Cnidarios = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))
Celenterados = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))
Poriferos = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))

# ******************************

# Reinos Plantae e Fungi
# ******************************
Plantae = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))
Fungi = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes = (8,8))

# ******************************

# Conecrivos e Quantificadores LTN
Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=6),semantics="exists")

formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError())

