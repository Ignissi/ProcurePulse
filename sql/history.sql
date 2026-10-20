-- История строго до текущего дня.
WITH history AS (
    SELECT
        contract_id,
        customer_inn,
        supplier_inn,
        COUNT(*) OVER buyer_window AS buyer_before,
        COUNT(*) OVER supplier_window AS supplier_before,
        COUNT(*) OVER pair_window AS pair_before
    FROM contracts
    WINDOW
        buyer_window AS (
            PARTITION BY customer_inn
            ORDER BY sign_date
            RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '1 day' PRECEDING
        ),
        supplier_window AS (
            PARTITION BY supplier_inn
            ORDER BY sign_date
            RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '1 day' PRECEDING
        ),
        pair_window AS (
            PARTITION BY customer_inn, supplier_inn
            ORDER BY sign_date
            RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '1 day' PRECEDING
        )
)
SELECT
    contract_id,
    CASE WHEN customer_inn <> '' THEN buyer_before END AS buyer_contracts,
    CASE WHEN supplier_inn <> '' THEN supplier_before END AS supplier_contracts,
    CASE WHEN customer_inn <> '' AND supplier_inn <> ''
        THEN pair_before END AS pair_contracts,
    CASE WHEN customer_inn <> '' AND supplier_inn <> ''
        THEN pair_before::DOUBLE PRECISION / NULLIF(buyer_before, 0) END AS pair_share
FROM history;
