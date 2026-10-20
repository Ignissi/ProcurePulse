-- По одной самой приоритетной закупке для каждой пары заказчик-поставщик.
WITH ranked AS (
    SELECT
        c.contract_id,
        c.customer_inn,
        c.supplier_inn,
        c.item_name,
        f.pair_contracts,
        f.pair_share,
        p.outcome_score,
        p.audit_priority,
        ROW_NUMBER() OVER (
            PARTITION BY c.customer_inn, c.supplier_inn
            ORDER BY p.audit_priority DESC, c.contract_id
        ) AS place_in_pair
    FROM contracts AS c
    JOIN features AS f ON f.contract_id = c.contract_id
    JOIN predictions AS p ON p.contract_id = c.contract_id
    WHERE c.customer_inn <> '' AND c.supplier_inn <> ''
)
SELECT *
FROM ranked
WHERE place_in_pair = 1
ORDER BY audit_priority DESC
LIMIT 100;
