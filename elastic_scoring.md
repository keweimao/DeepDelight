# Elastic Scoring

## BM25 Scoring

   The BM25 formula is given by:

   $$
   \text{score}(q, d) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}
   $$

   where:
   - \( q \) is a query
   - \( d \) is a document
   - \( f(q_i, d) \) is the frequency of term \( q_i \) in document \( d \)
   - \( |d| \) is the length of document \( d \)
   - \( \text{avgdl} \) is the average document length in the collection
   - \( k_1 \) and \( b \) are free parameters, usually chosen as \( k_1 = 1.2 \) and \( b = 0.75 \)
   - \( \text{IDF}(q_i) \) is the inverse document frequency of the term \( q_i \)

## Create BM25 from scratch in BM25


1. **Define the Index with Custom BM25 Similarity**:

   Create an index with a custom similarity setting using scripts that implement the BM25 formula.

   ```json
   PUT /my_index
   {
     "settings": {
       "number_of_shards": 1,
       "similarity": {
         "scripted_bm25": {
           "type": "scripted",
           "weight_script": {
             "source": """
               double k1 = 1.2;
               double b = 0.75;
               double idf = Math.log((field.docCount + 1.0) / (term.docFreq + 1.0)) + 1.0;
               return query.boost * idf;
             """
           },
           "script": {
             "source": """
               double k1 = 1.2;
               double b = 0.75;
               double tf = doc.freq;
               double docLength = doc.length;
               double avgdl = params.avgdl;
               double norm = (1 - b) + b * (docLength / avgdl);
               return weight * ((tf * (k1 + 1)) / (tf + k1 * norm));
             """,
             "params": {
               "avgdl": 100.0  // You should calculate and set this value based on your index
             }
           }
         }
       }
     },
     "mappings": {
       "properties": {
         "content": {
           "type": "text",
           "similarity": "scripted_bm25"
         }
       }
     }
   }
   ```

   In this example:
   - `"k1": 1.2` sets the term frequency saturation.
   - `"b": 0.75` sets the length normalization.
   - `idf` is computed as the inverse document frequency.
   - `tf` is the term frequency in the document.
   - `docLength` is the length of the current document.
   - `avgdl` is the average document length in the index, passed as a parameter.

2. **Index Documents**:

   Index your documents as usual. Elasticsearch will use the custom BM25 similarity for the specified fields.

   ```json
   POST /my_index/_doc/1
   {
     "content": "This is a sample document."
   }

   POST /my_index/_doc/2
   {
     "content": "Another example of a document with a different length."
   }
   ```

3. **Calculate Average Document Length**:

   Compute the average document length across the index. This value will be used in your custom BM25 script. Here, `avgdl` is set to 100.0 as an example; you should calculate this value based on your actual index data.

4. **Query the Index**:

   Perform a search query on the index. The custom BM25 similarity will be used automatically for the fields where it was specified in the mappings.

   ```json
   GET /my_index/_search
   {
     "query": {
       "match": {
         "content": "sample document"
       }
     }
   }
   ```

   The query will use the custom BM25 similarity parameters defined earlier.

### Explanation of Script Components

- **Index Settings**:
  - `similarity`: Defines custom similarity algorithms for the index.
  - `scripted_bm25`: The name of the custom similarity configuration.
  - `type`: The type of similarity algorithm to use, which is `scripted`.
  - `weight_script`: Computes the inverse document frequency (idf).
  - `script`: Computes the term frequency (tf) and applies the BM25 formula.

- **Mappings**:

  - `similarity`: Specifies which similarity algorithm to use for the field. Here, it is set to `scripted_bm25`.

### Considerations

- **Performance**: Custom script scoring can be slower than using built-in similarities due to the overhead of executing scripts.
- **Maintenance**: You need to manually manage and ensure that all required parameters (like average document length) are accurately provided and maintained.
- **Dynamic Parameters**: Ensure that the `avgdl` parameter is dynamically updated as the index grows.

