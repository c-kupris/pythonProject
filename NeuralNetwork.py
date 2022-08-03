import matplotlib.backend_managers
import matplotlib.pyplot as plot
import numpy
from Bio.codonalign import codonseq, CodonSeq
from Bio import Entrez
from Bio import SeqIO

Entrez.api_key = "my_apikey"
Entrez.email = "c.kupris@yahoo.com"


class NeuralNetwork:

    bias: float
    disease: float
    iterations: int
    some_gene: list[bytearray]
    some_mrna: str
    some_protein: str
    target: float
    weight: float

    matplotlib.use("pdf")

    def __init__(self, bias: float, iterations: int, weight: float):
        # define weights and biases
        numpy.random.seed(1)

        self.bias = bias
        self.iterations = iterations
        self.weight = weight

    @classmethod
    def set_some_gene(cls, gene: list[bytearray]) -> list[bytearray]:
        cls.some_gene = gene
        return gene

    @classmethod
    def forward_propagation(cls, weight, bias) -> []:
        a = cls.encode_gene(some_gene=cls.some_gene)
        return numpy.matmul(weight, a) + bias

    @classmethod
    def encode_gene(cls, some_gene) -> list[bytearray]:
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
        for x in some_gene:
            if nucleotide.keys().__contains__(x):
                z.append(nucleotide[x])
                print(nucleotide[x])
            print(z)
            cls.set_some_gene(gene=z)
            return z

    def encode_mrna(self, some_mrna) -> [bytearray]:
        self.some_mrna = some_mrna

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
        for x in some_mrna:
            if nucleotides.keys().__contains__(x):
                z.append(nucleotides[x])
                print(nucleotides[x])
            print(z)
            return z


    def encode_protein(self, protein) -> CodonSeq:
        self.some_protein = protein

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

        for x in protein:
            if nucleotides.keys().__contains__(x):
                if dictionary_of_proteins.__contains__(x):
                    z.append(nucleotides[x])
                    print(nucleotides[x])
            print(z)
            codon_seq = codonseq.CodonSeq(z)
            return codon_seq




    @classmethod
    def predict_disease(cls, weight, bias, some_gene) -> list:
        predictor = cls.encode_gene(some_gene=some_gene)
        weighted_network = predictor * weight
        biased_weighted_network = bias + weighted_network
        return biased_weighted_network

    # layer 1

    @classmethod
    def activation_function(cls, x) -> float:
        y = numpy.tanh(x)
        return y

    @classmethod
    def d_activation_function_dx(cls, x) -> float:
        y = 1 - (numpy.tanh(x) * numpy.tanh(x))
        return y

    @classmethod
    def layer1(cls, bias, weight) -> []:
        y = cls.predict_disease(cls, numpy.dot(weight, cls.encode_gene(cls.some_gene)), some_gene=cls.some_gene)
        x = cls.activation_function(y) + bias
        print(x)
        return x

    # layer 2

    @classmethod
    def sigmoid_activation_function(cls, x: float) -> float:
        y = numpy.exp(-x)
        a = 1 / (1 + y)
        return a

    @classmethod
    def d_sigmoid_dx(cls, x) -> float:
        y = 1 - cls.sigmoid_activation_function(x)
        return cls.sigmoid_activation_function(x) * y

    @classmethod
    def layer_2(cls, bias, weight) -> float:
        y: float = cls.layer1(cls, numpy.dot(weight, cls.encode_gene(cls.some_gene)))
        x = cls.sigmoid_activation_function(y) + bias
        print(x)
        return x

    # layer 3

    @classmethod
    def line(cls, x: float) -> float:
        return 5 * x

    @classmethod
    def d_line_dx(cls, x: float) -> float:
        return 5 * x

    @classmethod
    def layer_3(cls, bias, weight) -> float:
        y = cls.layer_2(cls, numpy.dot(cls.encode_gene(cls.some_gene), weight))
        a = cls.line(weight * y) + bias
        print(a)
        return a

    @classmethod
    def back_propagation(cls, weight, bias) -> float:
        #   Find ALL THE DERIVATIVES!!!!!!!
        # Layer 1

        if classmethod.__eq__(self=NeuralNetwork(bias=cls.bias, iterations=cls.iterations, weight=cls.weight),
                              o='layer1(cls, bias, weight)') and numpy.dot(weight, bias) > \
                numpy.minimum(numpy.tanh(weight),
                              numpy.tanh(bias)):
            dtanh_dx_pos = cls.d_activation_function_dx(cls) * cls.activation_function(cls)
            return dtanh_dx_pos

        elif classmethod.__eq__(self=NeuralNetwork(bias=cls.bias, iterations=cls.iterations, weight=cls.weight),
                                o='layer1(cls, bias, weight)') \
                and numpy.dot(weight, bias) < numpy.minimum(numpy.tanh(weight), numpy.tanh(bias)):
            dtanh_dx_neg = cls.d_activation_function_dx(cls) * cls.activation_function(cls)
            return dtanh_dx_neg

        # Layer 2

        elif classmethod.__eq__(self=NeuralNetwork(bias=cls.bias, iterations=cls.iterations, weight=cls.weight),
                                o='layer_2(cls, bias, weight)') \
                and numpy.dot(weight, bias) > numpy.minimum(cls.sigmoid_activation_function(weight)):
            dsigmoid_dx = cls.d_sigmoid_dx(cls.weight)
            return dsigmoid_dx * cls.activation_function(cls.weight)

        elif classmethod.__eq__(self=NeuralNetwork(bias=cls.bias, iterations=cls.iterations, weight=cls.weight),
                                o='layer_2(cls, bias, weight)') \
                and numpy.dot(weight, bias) < numpy.minimum(cls.sigmoid_activation_function(cls.weight)):
            dsigmoid_dx_neg = -cls.d_sigmoid_dx(cls.weight)
            return dsigmoid_dx_neg * cls.activation_function(cls.weight)

        # Layer 3

        elif classmethod.__eq__(self=NeuralNetwork(bias=cls.bias, iterations=cls.iterations, weight=cls.weight),
                                o='layer_3(cls, bias, weight)'
                                                                                 and numpy.dot(weight, bias)
                                                                                 > numpy.minimum
                                                                                     (
                                                                                     cls.line(cls.weight))):
            dline_dx = cls.d_line_dx(cls.weight)
            return dline_dx * cls.line(cls.weight)

        elif classmethod.__eq__(self=NeuralNetwork(bias=cls.bias, iterations=cls.iterations, weight=cls.weight),
                                o='layer_3(cls, bias, weight)' and numpy.dot(weight, bias)
                                  < numpy.minimum(cls.line(cls.weight))):
            dline_dx_neg = -cls.d_line_dx(cls.weight)
            return dline_dx_neg * cls.line(cls.weight)

        else:
            return 0

    # training the network

    @classmethod
    def train(cls):
        for _ in numpy.array(p_object=cls.iterations):
            if cls.layer_3(cls, weight=cls.weight) != cls.target:
                cls.back_propagation(weight=cls.weight, bias=cls.bias)
                _ = _ + 1

            elif cls.layer_3(cls, weight=cls.weight) == cls.target:
                return "My neural network works!"
            else:
                return 0

    @classmethod
    def search(cls, disease: float):
        handle = Entrez.esearch(db='pubmed', term=disease)
        cls.disease = disease
        cls.target = disease
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

            for some_mrna in SeqIO.index(filename=handle.read(), format='xml', alphabet='Single Letter Alphabet' or
                                                'Nucleotide Alphabet' or
                                                'Secondary Structure' or
                                                'Three Letter Protein' or
                                                'Protein Alphabet'):

                if index.contains(some_mrna):
                    cls.train()

                for protein in SeqIO.index(filename=handle.read(), format='xml', alphabet='RNA alphabet' or
                                                                                            'Single Letter Alphabet' or
                                                                                            'Nucleotide Alphabet' or
                                                                                            'Secondary Structure'):

                    if index.contains(protein):
                        cls.train()

        second_handle = Entrez.esearch(db='GenBank', term=disease)

        for second_gene in SeqIO.parse(handle=second_handle, format='xml', alphabet='DNA alphabet' or
                                               'RNA alphabet' or
                                               'Protein Alphabet' or
                                                'Single Letter Alphabet' or
                                                'Nucleotide Alphabet' or
                                                'Secondary Structure' or
                                                'Three Letter Protein'):

            for second_record in SeqIO.index(filename=second_handle.read(), format='xml', alphabet='DNA alphabet' or
                                               'RNA alphabet' or
                                               'Protein Alphabet' or
                                                'Single Letter Alphabet' or
                                                'Nucleotide Alphabet' or
                                                'Secondary Structure' or
                                                'Three Letter Protein'):

                Entrez.parse(handle=second_gene, validate=True, escape=True)
                second_gene_sequence = second_gene.seq

                if second_gene_sequence.contains(second_record):
                    cls.train()

                if second_gene_sequence.contains(cls.some_mrna) or second_record.contains(cls.some_mrna):
                    cls.train()

                if second_gene_sequence.contains(cls.some_protein) or second_record.contains(cls.some_protein):
                    cls.train()

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

                if third_gene_sequence.contains(third_record):
                    cls.train()

                if third_gene_sequence.contains(cls.some_mrna):
                    cls.train()

                if third_gene_sequence.contains(cls.some_protein):
                    cls.train()

        neural_network = NeuralNetwork(bias=0.02, iterations=1000, weight=0.05)

        numpy.random.seed(100)

        plot.figure()

        x = neural_network.iterations
        print(x)
        plot.xlabel = "Iterations"
        plot.xscale = 1
        plot.xticks(ticks=numpy.arrange(1, 2, step=1))

        y = neural_network.target
        print(y)

        plot.ylabel = "Target"
        plot.yscale = 1
        plot.yticks(ticks=numpy.arrange(0, 1, step=0.1))

        plot.grid()
        plot.style.use(style='bmh')

        legend = plot.legend
        legend.loc = 3

        plot.title = "Neural Network Target as a Function of Iterations"

        plot.plot(x, y)

        plot.savefig("Iterations_vs_Target.pdf")
