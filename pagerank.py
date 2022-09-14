import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    print(corpus)
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            
            pages[filename] = set(links) - {filename}
    
    # print(pages)
    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )
    # print("------")
    # print(pages)
    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_dist= dict()

    page_count_in_corpus = len(corpus)

    if len(corpus[page]) > 0:
        random_prob = (1 - damping_factor) / len(corpus)
        
        eq_prob = damping_factor / len(corpus[page])

        for key in corpus.keys():
            if key not in corpus[page]:
                prob_dist[key] = random_prob
            else:
                prob_dist[key] = eq_prob + random_prob
    else:
        for key in corpus.keys():
            prob_dist[key] = 1 / page_count_in_corpus
    # print(prob_dist)
    return prob_dist

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank_dict = dict()

    selected_page = random.choice(list(corpus))

    for i in range(n):
        prob = transition_model(corpus, selected_page, damping_factor)

        selected_page = random.choices(list(prob), weights=prob.values(), k=1).pop()

        if selected_page in pagerank_dict:
            pagerank_dict[selected_page] += 1
        else:
            pagerank_dict[selected_page] = 1

    for page in pagerank_dict:
        pagerank_dict[page] = pagerank_dict[page] / n
        
    return pagerank_dict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    starting_pagerank = dict()

    current_pagerank = dict()

    for page in corpus:
        starting_pagerank[page] = 1 / len(corpus)

    while True:
        for page in corpus:
            temp_prob = 0
            for ref_page in corpus:
                if page in corpus[ref_page]:
                    temp_prob += (starting_pagerank[ref_page] / len(corpus[ref_page]))
                if len(corpus[ref_page]) == 0:
                    temp_prob += (starting_pagerank[ref_page]) / len(corpus)
            temp_prob *= damping_factor
            temp_prob += (1 - damping_factor) / len(corpus)

            current_pagerank[page] = temp_prob

        difference = max([abs(current_pagerank[x] - starting_pagerank[x]) for x in starting_pagerank])
        if difference < 0.001:
            break
        else:
            starting_pagerank = current_pagerank.copy()

    return starting_pagerank


if __name__ == "__main__":
    main()
