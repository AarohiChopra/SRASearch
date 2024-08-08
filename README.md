This was an experimental project which aims to provide a platform for querying SRA using natural language processing. 
We are utilizing contextual information by using a technique called word embeddings. 
For each submission_accession, we transform the columns (this includes abstract, metadata etc) into a vector of 700 dimensions. 
Of all these vectors, we build indexes and perform similarity search. 
