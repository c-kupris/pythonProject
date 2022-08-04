import matplotlib.pyplot as plt
import numpy
from Bio import Entrez
from Bio import SeqIO

Entrez.api_key = "my_apikey"
Entrez.email = "c.kupris@yahoo.com"


class NeuralNetwork:
    bias: bytes
    disease: float
    iterations: int
    some_gene: str
    some_mrna: str
    some_protein: str
    target: float
    weight: float

    def __init__(self, bias: bytes, iterations: int, weight: float, target: float):
        # define weights and biases
        numpy.random.seed(1)

        self.bias = bias
        self.iterations = iterations
        self.target = target
        self.weight = weight

    @classmethod
    def forward_propagation(cls, weight, bias) -> []:
        a = cls.encode_gene(some_gene=cls.some_gene)
        return numpy.matmul(weight, a) + bias

    @classmethod
    def encode_gene(cls, some_gene) -> list:

        cls.some_gene = some_gene

        # Assign each base a bit of information

        a = [0, 0]
        t = [0, 1]
        g = [1, 0]
        c = [1, 1]

        nucleotide: dict = {
            "A": a,
            "T": t,
            "G": g,
            "C": c,
        }

        z = [bytearray]

        #  for each of the letters in the genetic sequence...
        for gene in some_gene:
            if nucleotide.keys().__contains__(gene):
                z.append(nucleotide[gene])
                print(nucleotide[gene])
            print(z)
            return z

    @classmethod
    def encode_mrna(cls, some_mrna) -> list:
        cls.some_mrna = some_mrna

        # Assign each base a bit of information

        a = [0, 0]
        u = [0, 1]
        g = [1, 0]
        c = [1, 1]

        nucleotides: dict = {
            "A": a,
            "U": u,
            "G": g,
            "C": c,
        }

        z = [bytearray]

        #  for each of the letters in the genetic sequence...
        for mrna in some_mrna:
            if nucleotides.keys().__contains__(mrna):
                z.append(nucleotides[mrna])
                print(nucleotides[mrna])
            print(z)
            return z

    @classmethod
    def encode_protein(cls, protein) -> list:
        cls.some_protein = protein

        # Assign each base a bit of information

        a = [0, 0]
        u = [0, 1]
        g = [1, 0]
        c = [1, 1]

        nucleotides: dict = {
            "A": a,
            "U": u,
            "G": g,
            "C": c,
        }

        dictionary_of_proteins: dict = {
            "Alanine": "GCA" or "GCC" or "GCG" or "GCU",
            "Arginine": "CGA" or "CGC" or "CGG" or "CGU",
            "Asparagine": "AAC" or "AAU",
            "Aspartic Acid": "GAC" or "GAU",
            "Cysteine": "UGC" or "UGU",
            "Glutamic Acid": "GAA" or "GAG",
            "Glutamine": "CAA" or "CAG",
            "Glycine": "GGA" or "GGC" or "GGG" or "GGU",
            "Histidine": "CAC" or "CAU",
            "Isoleucine": "AUA" or "AUC" or "AUU",
            "Leucine": "UUA" or "UUG",
            "Lysine": "AAA" or "AAG",
            "Methionine": "AUG",
            "Phenylalanine": "UUC" or "UUU",
            "Proline": "CCA" or "CCC" or "CCG" or "CCU",
            # Find the nucleotide sequence for "Pyrrolysine" : ,
            # Find the nucleotide sequence for "Selenocysteine" : ,
            "Serine": "AGC" or "AGU",
            "Threonine": "ACA" "ACC" "ACG" "ACU",
            "Tryptophan": "UGG",
            "Tyrosine": "UAC" or "UAU",
            "Valine": "GUA" or "GUC" or "GUG" or "GUU",
            "Start": "AUG",
            "Stop": "UAA" or "UAG" or "UGA",
        }

        z = []

        for _x in protein:
            if nucleotides.keys().__contains__(_x):
                if dictionary_of_proteins.__contains__(_x):
                    z.append(nucleotides[_x])
                    print(nucleotides[_x])
            print(z)
            return z

    @classmethod
    def predict_disease(cls, weight, bias, some_gene, some_mrna, some_protein) -> float:
        predictor_dna = cls.encode_gene(some_gene=some_gene)
        predictor_rna = cls.encode_mrna(some_mrna=some_mrna)
        predictor_protein = cls.encode_protein(protein=some_protein)

        weighted_network = (predictor_rna * weight) + (predictor_dna * weight) + (predictor_protein * weight)
        biased_weighted_network = list(bytearray(bias)) + weighted_network
        return biased_weighted_network

    # layer 1

    @classmethod
    def activation_function(cls, _x) -> float:
        _y = numpy.tanh(_x)
        return _y

    @classmethod
    def d_activation_function_dx(cls, _x) -> float:
        _y = 1 - (numpy.tanh(_x) * numpy.tanh(_x))
        return _y

    @classmethod
    def layer1(cls, bias, weight) -> []:
        _y = cls.predict_disease(cls, numpy.dot(weight, cls.encode_gene(cls.some_gene)), some_gene=cls.some_gene,
                                 some_mrna=cls.some_mrna, some_protein=cls.some_protein)
        _x = cls.activation_function(_y) + bias
        print(_x)
        return _x

    # layer 2

    @classmethod
    def sigmoid_activation_function(cls, _x: float) -> float:
        _y = numpy.exp(-_x)
        a = 1 / (1 + _y)
        return a

    @classmethod
    def d_sigmoid_dx(cls, _x) -> float:
        _y = 1 - cls.sigmoid_activation_function(_x)
        return cls.sigmoid_activation_function(_x) * _y

    @classmethod
    def layer_2(cls, bias, weight) -> float:
        _y: float = cls.layer1(cls, numpy.dot(weight, cls.encode_gene(cls.some_gene)))
        _x = cls.sigmoid_activation_function(_y) + bias
        print(_x)
        return _x

    # layer 3

    @classmethod
    def line(cls, _x: float) -> float:
        return 5 * _x

    @classmethod
    def d_line_dx(cls, _x: float) -> float:
        return 5 * _x

    @classmethod
    def layer_3(cls, bias, weight) -> float:
        _y: float = cls.layer_2(cls, numpy.dot(weight, cls.encode_gene(cls.some_gene)))
        a = cls.line(weight * _y) + bias
        print(a)
        return a

    @classmethod
    def back_propagation(cls, weight, bias) -> float:
        #   Find ALL THE DERIVATIVES!!!!!!!
        # Layer 1

        if classmethod.__eq__(self=NeuralNetwork(bias=cls.bias, iterations=cls.iterations, target=cls.target, weight=cls.weight),
                              o='layer1(cls, bias, weight)') and numpy.dot(weight, bias) > \
                numpy.minimum(numpy.tanh(weight),
                              numpy.tanh(bias)):
            dtanh_dx_pos = \
                1 / (numpy.cosh(cls.predict_disease(weight=cls.weight, bias=cls.bias, some_gene=cls.some_gene,
                                                    some_mrna=cls.some_mrna, some_protein=cls.some_protein))) * \
                1 / (numpy.cosh(cls.predict_disease(weight=cls.weight, bias=cls.bias, some_gene=cls.some_gene,
                                                    some_mrna=cls.some_mrna, some_protein=cls.some_protein)))
            return dtanh_dx_pos * cls.predict_disease(weight=cls.weight, bias=cls.bias, some_gene=cls.some_gene,
                                                      some_mrna=cls.some_mrna, some_protein=cls.some_protein)

        elif classmethod.__eq__(self=NeuralNetwork(bias=cls.bias, iterations=cls.iterations, target=cls.target, weight=cls.weight),
                                o='layer1(cls, bias, weight)') \
                and numpy.dot(weight, bias) < numpy.minimum(numpy.tanh(weight), numpy.tanh(bias)):
            dtanh_dx_neg = -1 / (numpy.cosh(
                cls.predict_disease(weight=cls.weight, bias=cls.bias, some_gene=cls.some_gene, some_mrna=cls.some_mrna,
                                    some_protein=cls.some_protein))) * \
                           1 / (numpy.cosh(cls.predict_disease(weight=cls.weight,
                                                               bias=cls.bias,
                                                               some_gene=cls.some_gene,
                                                               some_mrna=cls.some_mrna,
                                                               some_protein=cls.some_protein)))
            return dtanh_dx_neg * cls.predict_disease(weight=cls.weight, bias=cls.bias, some_gene=cls.some_gene,
                                                      some_mrna=cls.some_mrna, some_protein=cls.some_protein)

        # Layer 2

        elif classmethod.__eq__(self=NeuralNetwork(bias=cls.bias, iterations=cls.iterations, target=cls.target, weight=cls.weight),
                                o='layer_2(cls, bias, weight)') \
                and numpy.dot(weight, bias) > numpy.minimum(cls.sigmoid_activation_function(weight)):
            dsigmoid_dx = cls.d_sigmoid_dx(cls.weight)
            return dsigmoid_dx * cls.predict_disease(weight=cls.weight, bias=cls.bias, some_gene=cls.some_gene,
                                                     some_mrna=cls.some_mrna, some_protein=cls.some_protein)

        elif classmethod.__eq__(self=NeuralNetwork(bias=cls.bias, iterations=cls.iterations, target=cls.target, weight=cls.weight),
                                o='layer_2(cls, bias, weight)') \
                and numpy.dot(weight, bias) < numpy.minimum(cls.sigmoid_activation_function(cls.weight)):
            dsigmoid_dx_neg = -cls.d_sigmoid_dx(cls.weight)
            return dsigmoid_dx_neg * cls.predict_disease(weight=cls.weight, bias=cls.bias, some_gene=cls.some_gene,
                                                         some_mrna=cls.some_mrna, some_protein=cls.some_protein)

        # Layer 3

        elif classmethod.__eq__(self=NeuralNetwork(bias=cls.bias, iterations=cls.iterations, target=cls.target, weight=cls.weight),
                                o='layer_3(cls, bias, weight)'
                                  and numpy.dot(weight, bias)
                                  > numpy.minimum
                                      (
                                      cls.line(cls.weight))):
            dline_dx = cls.d_line_dx(cls.weight)
            return dline_dx * cls.line(cls.weight) * \
                   cls.predict_disease(weight=cls.weight, bias=cls.bias, some_gene=cls.some_gene,
                                       some_mrna=cls.some_mrna, some_protein=cls.some_protein)

        elif classmethod.__eq__(self=NeuralNetwork(bias=cls.bias, iterations=cls.iterations, target=cls.target, weight=cls.weight),
                                o='layer_3(cls, bias, weight)' and numpy.dot(weight, bias)
                                  < numpy.minimum(cls.line(cls.weight))):
            dline_dx_neg = -cls.d_line_dx(cls.weight)
            return dline_dx_neg * cls.line(cls.weight) * \
                   cls.predict_disease(weight=cls.weight, bias=cls.bias, some_gene=cls.some_gene,
                                       some_mrna=cls.some_mrna, some_protein=cls.some_protein)

        else:
            return 0

    @classmethod
    def update(cls, target_to_update: float) -> float:

        for _ in numpy.arange(start=0, stop=cls.iterations, step=1):
            weighted_gene = cls.weight * cls.encode_gene(cls.some_gene)[_]
            weighted_mrna = cls.weight * cls.encode_mrna(cls.some_mrna)[_]
            weighted_protein = cls.weight * cls.encode_protein(cls.some_protein)[_]
            target_to_update = (cls.weight * weighted_gene) + (cls.weight * weighted_mrna) + \
                               (cls.weight * weighted_protein)
            print(target_to_update)
            cls.target = target_to_update
            return target_to_update

    # training the network

    @classmethod
    def train(cls) -> float:
        for _ in numpy.array(p_object=cls.iterations):
            if cls.layer_3(cls, weight=cls.weight) != cls.target:
                cls.back_propagation(weight=cls.weight, bias=cls.bias)
                cls.update(target_to_update=cls.target)
                _ = _ + 1

            elif cls.layer_3(cls, weight=cls.weight) == cls.target:
                return cls.target
            else:
                return 0

    @classmethod
    def search(cls, disease: float):
        handle = Entrez.esearch(db='pubmed', term=disease)
        cls.disease = disease
        for index in SeqIO.parse(handle, format='xml', alphabet='DNA alphabet' or
                                                                'RNA alphabet' or
                                                                'Protein Alphabet' or
                                                                'Single Letter Alphabet' or
                                                                'Nucleotide Alphabet' or
                                                                'Secondary Structure' or
                                                                'Three Letter Protein').records:

            for gene in SeqIO.index(filename=handle.read(), format='xml', alphabet='DNA alphabet' or
                                                                                   'RNA alphabet' or
                                                                                   'Protein Alphabet' or
                                                                                   'Single Letter Alphabet' or
                                                                                   'Nucleotide Alphabet' or
                                                                                   'Secondary Structure' or
                                                                                   'Three Letter Protein'):
                Entrez.parse(handle=gene, validate=True, escape=True)
                cls.some_gene = gene
                cls.update(cls.target)

            for some_mrna in SeqIO.index(filename=handle.read(), format='xml', alphabet='Single Letter Alphabet' or
                                                                                        'Nucleotide Alphabet' or
                                                                                        'Secondary Structure' or
                                                                                        'Three Letter Protein' or
                                                                                        'Protein Alphabet'):

                if index.contains(some_mrna):
                    cls.some_mrna = some_mrna
                    cls.train()
                    cls.update(cls.target)

                for protein in SeqIO.index(filename=handle.read(), format='xml', alphabet='RNA alphabet' or
                                                                                          'Single Letter Alphabet' or
                                                                                          'Nucleotide Alphabet' or
                                                                                          'Secondary Structure'):

                    if index.contains(protein):
                        cls.some_protein = protein
                        cls.train()
                        cls.update(cls.target)

        second_handle = Entrez.esearch(db='GenBank', term=disease)

        for second_gene in SeqIO.parse(handle=second_handle, format='xml', alphabet='DNA alphabet' or
                                                                                    'RNA alphabet' or
                                                                                    'Protein Alphabet' or
                                                                                    'Single Letter Alphabet' or
                                                                                    'Nucleotide Alphabet' or
                                                                                    'Secondary Structure' or
                                                                                    'Three Letter Protein'):

            for second_record in SeqIO.index(filename=second_handle.read(), format='xml', alphabet=('DNA alphabet' or
                                                                                                    'RNA alphabet' or
                                                                                                    'Protein Alphabet'
                                                                                                    or 'Single Letter '
                                                                                                       'Alphabet') or
                                                                                                   'Nucleotide Alphabet' or
                                                                                                   'Secondary Structure' or
                                                                                                   'Three Letter Protein'):

                Entrez.parse(handle=second_gene, validate=True, escape=True)
                second_gene_sequence = second_gene.seq
                cls.some_gene = second_gene_sequence
                cls.update(cls.target)

                if second_gene_sequence.contains(second_record):
                    cls.train()
                    cls.update(cls.target)

                if second_gene_sequence.contains(cls.some_mrna) or second_record.contains(cls.some_mrna):
                    cls.train()
                    cls.update(cls.target)

                if second_gene_sequence.contains(cls.some_protein) or second_record.contains(cls.some_protein):
                    cls.train()
                    cls.update(cls.target)

        third_handle = Entrez.esearch(db='NIH', term=disease)

        for third_gene in SeqIO.parse(handle=third_handle,
                                      format='xml',
                                      alphabet='DNA alphabet' or
                                               'RNA alphabet' or
                                               'Protein Alphabet' or
                                               'Single Letter Alphabet' or
                                               'Nucleotide Alphabet' or
                                               'Secondary Structure' or
                                               'Three Letter Protein'):

            for third_record in SeqIO.index(format='xml', filename='xml', alphabet='DNA alphabet' or
                                                                                   'RNA alphabet' or
                                                                                   'Protein Alphabet' or
                                                                                   'Single Letter Alphabet' or
                                                                                   'Nucleotide Alphabet' or
                                                                                   'Secondary Structure' or
                                                                                   'Three Letter Protein'):

                Entrez.parse(third_gene, validate=True, escape=True)
                third_gene_sequence = third_gene.seq
                cls.some_gene = third_gene_sequence
                cls.update(cls.target)

                if third_gene_sequence.contains(third_record):
                    cls.train()
                    cls.update(cls.target)

                if third_gene_sequence.contains(cls.some_mrna):
                    cls.train()
                    cls.update(cls.target)

                if third_gene_sequence.contains(cls.some_protein):
                    cls.train()
                    cls.update(cls.target)


neural_network = NeuralNetwork(bias=bytes(2), iterations=10, target=0.1, weight=5)

x = numpy.arange(start=0, stop=neural_network.iterations, step=1)

print(x)

y = numpy.arange(start=0, stop=neural_network.target, step=0.01)

print(y)

plt.plot(x, y)

plt.show()

print("done")
