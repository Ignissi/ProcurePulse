-- Пары с тремя и более контрактами.
SELECT
    customer_inn,
    supplier_inn,
    COUNT(*) AS contracts_count,
    MIN(sign_date) AS first_contract,
    MAX(sign_date) AS last_contract
FROM contracts
WHERE customer_inn <> '' AND supplier_inn <> ''
GROUP BY customer_inn, supplier_inn
HAVING COUNT(*) >= 3
ORDER BY contracts_count DESC;

-- Время между последовательными контрактами одной пары.
SELECT
    contract_id,
    customer_inn,
    supplier_inn,
    sign_date,
    LAG(sign_date) OVER (
        PARTITION BY customer_inn, supplier_inn
        ORDER BY sign_date, contract_id
    ) AS previous_date
FROM contracts
WHERE customer_inn <> '' AND supplier_inn <> '';
