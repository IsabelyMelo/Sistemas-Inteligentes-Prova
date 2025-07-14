"""Questão 01 – Desenvolva um algoritmo FP-Growth para resolver o problema
Market_Basket_Optimisation (disponível juntamente com a prova). Utilize um suporte
mínimo de 300 e confiança mínima de 0,3. Apresente os dados analisados, destacando:
• Os itens que mais aparecem individualmente nas transações;
• Os conjuntos de itens mais frequentes que ocorrem juntos."""

import os
import pandas as pd
import matplotlib.pyplot as plt

# carregar transacoes
def load_transactions(filepath):
    df = pd.read_csv(filepath, header=None)
    transactions = df.apply(lambda row: [item for item in row if pd.notna(item)], axis=1).tolist()
    return transactions

# estrutura de no da arvore
class FPTreeNode:
    def __init__(self, item_name, parent_node):
        self.name = item_name
        self.count = 1
        self.parent = parent_node
        self.children = {}
        self.link = None

# contagem de itens
def count_items(transactions):
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1
    return item_counts

# criacao da FP-Tree
def build_tree(transactions, min_support):
    item_counts = count_items(transactions)
    item_counts = {item: count for item, count in item_counts.items() if count >= min_support}
    if not item_counts:
        return None, None

    header_table = {item: [count, None] for item, count in item_counts.items()}
    root = FPTreeNode(None, None)

    for transaction in transactions:
        filtered = [item for item in transaction if item in item_counts]
        sorted_items = sorted(filtered, key=lambda item: (-item_counts[item], item))
        current = root
        for item in sorted_items:
            if item in current.children:
                current.children[item].count += 1
            else:
                new_node = FPTreeNode(item, current)
                current.children[item] = new_node
                if header_table[item][1] is None:
                    header_table[item][1] = new_node
                else:
                    node = header_table[item][1]
                    while node.link:
                        node = node.link
                    node.link = new_node
            current = current.children[item]

    return root, header_table

# extracao de padroes frequentes
def extract_patterns(header_table, min_support):
    def ascend_tree(node):
        path = []
        while node.parent and node.parent.name:
            node = node.parent
            path.append(node.name)
        return path[::-1]

    patterns = []
    for item in sorted(header_table, key=lambda i: header_table[i][0]):
        support = header_table[item][0]
        pattern_base = []
        node = header_table[item][1]
        while node:
            path = ascend_tree(node)
            if path:
                pattern_base.append((path, node.count))
            node = node.link

        conditional_input = []
        for path, count in pattern_base:
            for _ in range(count):
                conditional_input.append(path[:])

        subtree, sub_header = build_tree(conditional_input, min_support)
        suffix = frozenset([item])
        patterns.append((suffix, support))
        if sub_header:
            sub_patterns = extract_patterns(sub_header, min_support)
            for pattern, count in sub_patterns:
                patterns.append((pattern.union(suffix), count))
    return patterns

# gera combinações de tamanho r dos itens
def custom_combinations(items, r):
    items = list(items)
    n = len(items)
    result = []

    def generate(start, comb):
        if len(comb) == r:
            result.append(tuple(comb))
            return
        for i in range(start, n):
            generate(i + 1, comb + [items[i]])

    generate(0, [])
    return result

# geracao de regras de associacao
def generate_rules(frequent_itemsets, min_confidence):
    rules = []
    freq_dict = {frozenset(items): support for items, support in frequent_itemsets}
    for items, support in frequent_itemsets:
        if len(items) < 2:
            continue
        items = frozenset(items)
        for i in range(1, len(items)):
            for antecedent in custom_combinations(items, i):
                antecedent = frozenset(antecedent)
                consequent = items - antecedent
                if freq_dict.get(antecedent, 0) > 0:
                    confidence = support / freq_dict[antecedent]
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, support, confidence))
    return rules

# graficos
def plot_graph(data, title, xlabel, filename):
    labels, values = zip(*data)

    os.makedirs("questao-1/graficos", exist_ok=True)

    plt.figure(figsize=(12, 6))
    bars = plt.barh(labels[::-1], values[::-1], color='seagreen')
    plt.xlabel(xlabel)
    plt.title(title)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 5, bar.get_y() + bar.get_height() / 2, str(int(width)), va='center')
    plt.subplots_adjust(left=0.2)
    plt.savefig(f"questao-1/graficos/{filename}", dpi=300)
    plt.close()



if __name__ == "__main__":
    filepath = "questao-1/Market_Basket_Optimisation.csv"
    min_support = 300
    min_confidence = 0.3

    transactions = load_transactions(filepath)
    print(f"Total de transações carregadas: {len(transactions)}")

    tree, header_table = build_tree(transactions, min_support)
    frequent_itemsets = extract_patterns(header_table, min_support)
    rules = generate_rules(frequent_itemsets, min_confidence)

    individual_items = sorted([(list(fs)[0], sup) for fs, sup in frequent_itemsets if len(fs) == 1], key=lambda x: -x[1])
    multi_items = sorted([(fs, sup) for fs, sup in frequent_itemsets if len(fs) > 1], key=lambda x: (-x[1], -len(x[0])))
    sorted_rules = sorted(rules, key=lambda x: (-x[3], -x[2]))

    print("\nItens que mais aparecem individualmente nas transações (Suporte ≥ 300):")
    for item, support in individual_items[:20]:
        print(f"{item}: {support}")

    print("\nConjuntos de itens mais frequentes que ocorrem juntos (Suporte ≥ 300):")
    for itemset, support in multi_items[:20]:
        print(f"{', '.join(itemset)}: {support}")

    print("\nRegras de Associação (Confiança ≥ 0.3):")
    for antecedent, consequent, support, confidence in sorted_rules[:20]:
        print(f"{', '.join(antecedent)} → {', '.join(consequent)} (Suporte: {support}, Confiança: {confidence:.2f})")

    plot_graph(individual_items[:15], 
                        "Itens Mais Frequentes (Suporte ≥ 300)", 
                        "Suporte", 
                        "itens_frequentes.png")

    plot_graph([(f"{', '.join(itemset)}", sup) for itemset, sup in multi_items[:15]],
                        "Conjuntos Frequentes (Suporte ≥ 300)", 
                        "Suporte", 
                        "conjuntos_frequentes.png")

    plot_graph([(f"{', '.join(a)} → {', '.join(c)}", conf) for a, c, _, conf in sorted_rules[:15]],
                        "Regras de Associação (Confiança ≥ 0.3)", 
                        "Confiança", 
                        "regras_associacao.png")
