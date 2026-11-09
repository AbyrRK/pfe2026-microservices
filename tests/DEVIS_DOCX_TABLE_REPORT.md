# Devis_32.docx — table chunking comparison

Table lines detected: 8

| Strategy | Total chunks | Broken table rows |
|---|---|---|
| Naive recursive | 20 | 6 |
| create_chunks | 13 | 0 |

## Naive chunks

- `'Société T.P et services\n\nARIANA CENTRE BLOC C BUREAU 408'`
- `'T.V.A 1823212 V /A /M 000\n\nTEL 98 940 912 FAX 73 252 034'`
- `'Tunis le, 12/04/2026\n\nDevis N° 32'`
- `"Travaux de branchement de conduite d'eaux usée\n\nAriana"`
- `'Arrêter la présente facture a la somme de trente-huit'`
- `'milles sept cent quatre cent quatre-vingt-quatorze dinars'`
- `'Signature\n\n--- EXTRACTED TABLES ---'`
- `'[Table 1]'`
- `'| Désignation | Unité | Quantité | Prix U HT | Prix TOTAL |'`
- `'| Conduite PVC ∅ 250, y compris fouille et remblaiement et'`
- `'mise a la cote | ML | 80 | 270 ,000 | 21.600,000 |'`
- `'| Regard de visite ∅ 1000, y compris tampon lourd type Onas'`
- `'| U | 4 | 1.800,000 | 7.200,000 |'`
- `'| Caillage de la route existante et rattrapage de la couche'`
- `"d'enrobe | M2 | 15 | 120 ,000 | 1.800,000 |"`
- `'| Branchement au réseaux Onas existent | U | 1 | 2000,000 |'`
- `'|'`
- `'| | Prix total HT | 32.600,000 | | |'`
- `'| | TVA 19% | 6.194,000 | | |'`
- `'| | Total TTC | 38.794 ,000 | | |'`

## create_chunks output

- `'Société T.P et services\n\nARIANA CENTRE BLOC C BUREAU 408'`
- `'T.V.A 1823212 V /A /M 000\n\nTEL 98 940 912 FAX 73 252 034'`
- `'Tunis le, 12/04/2026\n\nDevis N° 32'`
- `"Travaux de branchement de conduite d'eaux usée\n\nAriana"`
- `'Arrêter la présente facture a la somme de trente-huit'`
- `'milles sept cent quatre cent quatre-vingt-quatorze dinars'`
- `'Signature\n\n--- EXTRACTED TABLES ---\n\n[Table 1]'`
- `'| Désignation | Unité | Quantité | Prix U HT | Prix TOTAL |'`
- `'| Désignation | Unité | Quantité | Prix U HT | Prix TOTAL |\n| Conduite PVC ∅ 250, y compris fouille et remblaiement et mise a la cote | ML | 80 | 270 ,000 | 21.600,000 |'`
- `'| Désignation | Unité | Quantité | Prix U HT | Prix TOTAL |\n| Regard de visite ∅ 1000, y compris tampon lourd type Onas | U | 4 | 1.800,000 | 7.200,000 |'`
- `"| Désignation | Unité | Quantité | Prix U HT | Prix TOTAL |\n| Caillage de la route existante et rattrapage de la couche d'enrobe | M2 | 15 | 120 ,000 | 1.800,000 |"`
- `'| Désignation | Unité | Quantité | Prix U HT | Prix TOTAL |\n| Branchement au réseaux Onas existent | U | 1 | 2000,000 | |\n| | Prix total HT | 32.600,000 | | |'`
- `'| Désignation | Unité | Quantité | Prix U HT | Prix TOTAL |\n| | TVA 19% | 6.194,000 | | |\n| | Total TTC | 38.794 ,000 | | |'`
