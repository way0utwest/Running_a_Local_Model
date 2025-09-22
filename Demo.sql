USE AdventureWorksLT
GO
CREATE OR ALTER PROCEDURE Ask_A_Question
  @Question NVARCHAR(MAX)
AS
DECLARE @search_text NVARCHAR(MAX) = @Question
DECLARE @search_vector VECTOR(768) = AI_GENERATE_EMBEDDINGS(@search_text USE MODEL ollama);

SELECT
    t.ProductID,
    t.chunk,
    s.distance,
    p.ListPrice,
	pd.Description
FROM vector_search(
    table = [SalesLT].[ProductEmbeddings] AS t,
    column = [embeddings],
    similar_to = @search_vector,
    metric = 'cosine',
    top_n = 10
) AS s
JOIN [SalesLT].[Product] p ON t.ProductID = p.ProductID
INNER JOIN SalesLT.vPRoductAndDescription pd ON p.ProductID = pd.ProductID AND Culture = 'en'
ORDER BY s.distance;
GO
EXEC Ask_A_Question 'I want the bike that is good for women'
GO
EXEC Ask_A_Question 'I want a blue bike and dont want to spend a lot'
GO
SELECT * FROM SalesLT.vPRoductAndDescription