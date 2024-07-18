import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


# Step 1: Parse the provided data into a structured format
data = [
    {"Bit Range": 1, "Private Key": "0000000000000000000000000000000000000000000000000000000000000001", "Address": "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH"},
    {"Bit Range": 2, "Private Key": "0000000000000000000000000000000000000000000000000000000000000003", "Address": "1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb"},
    {"Bit Range": 3, "Private Key": "0000000000000000000000000000000000000000000000000000000000000007", "Address": "19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA"},
    {"Bit Range": 4, "Private Key": "0000000000000000000000000000000000000000000000000000000000000008", "Address": "1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e"},
    {"Bit Range": 5, "Private Key": "0000000000000000000000000000000000000000000000000000000000000015", "Address": "1E6NuFjCi27W5zoXg8TRdcSRq84zJeBW3k"},
    {"Bit Range": 6, "Private Key": "0000000000000000000000000000000000000000000000000000000000000031", "Address": "1PitScNLyp2HCygzadCh7FveTnfmpPbfp8"},
    {"Bit Range": 7, "Private Key": "000000000000000000000000000000000000000000000000000000000000004c", "Address": "1McVt1vMtCC7yn5b9wgX1833yCcLXzueeC"},
    {"Bit Range": 8, "Private Key": "00000000000000000000000000000000000000000000000000000000000000e0", "Address": "1M92tSqNmQLYw33fuBvjmeadirh1ysMBxK"},
    {"Bit Range": 9, "Private Key": "00000000000000000000000000000000000000000000000000000000000001d3", "Address": "1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV"},
    {"Bit Range": 10, "Private Key": "0000000000000000000000000000000000000000000000000000000000000202", "Address": "1LeBZP5QCwwgXRtmVUvTVrraqPUokyLHqe"},
    {"Bit Range": 11, "Private Key": "0000000000000000000000000000000000000000000000000000000000000483", "Address": "1PgQVLmst3Z314JrQn5TNiys8Hc38TcXJu"},
    {"Bit Range": 12, "Private Key": "0000000000000000000000000000000000000000000000000000000000000a7b", "Address": "1DBaumZxUkM4qMQRt2LVWyFJq5kDtSZQot"},
    {"Bit Range": 13, "Private Key": "0000000000000000000000000000000000000000000000000000000000001460", "Address": "1Pie8JkxBT6MGPz9Nvi3fsPkr2D8q3GBc1"},
    {"Bit Range": 14, "Private Key": "0000000000000000000000000000000000000000000000000000000000002930", "Address": "1ErZWg5cFCe4Vw5BzgfzB74VNLaXEiEkhk"},
    {"Bit Range": 15, "Private Key": "00000000000000000000000000000000000000000000000000000000000068f3", "Address": "1QCbW9HWnwQWiQqVo5exhAnmfqKRrCRsvW"},
    {"Bit Range": 16, "Private Key": "000000000000000000000000000000000000000000000000000000000000c936", "Address": "1BDyrQ6WoF8VN3g9SAS1iKZcPzFfnDVieY"},
    {"Bit Range": 17, "Private Key": "000000000000000000000000000000000000000000000000000000000001764f", "Address": "1HduPEXZRdG26SUT5Yk83mLkPyjnZuJ7Bm"},
    {"Bit Range": 18, "Private Key": "000000000000000000000000000000000000000000000000000000000003080d", "Address": "1GnNTmTVLZiqQfLbAdp9DVdicEnB5GoERE"},
    {"Bit Range": 19, "Private Key": "000000000000000000000000000000000000000000000000000000000005749f", "Address": "1NWmZRpHH4XSPwsW6dsS3nrNWfL1yrJj4w"},
    {"Bit Range": 20, "Private Key": "00000000000000000000000000000000000000000000000000000000000d2c55", "Address": "1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum"},
    {"Bit Range": 21, "Private Key": "00000000000000000000000000000000000000000000000000000000001ba534", "Address": "14oFNXucftsHiUMY8uctg6N487riuyXs4h"},
    {"Bit Range": 22, "Private Key": "00000000000000000000000000000000000000000000000000000000002de40f", "Address": "1CfZWK1QTQE3eS9qn61dQjV89KDjZzfNcv"},
    {"Bit Range": 23, "Private Key": "0000000000000000000000000000000000000000000000000000000000556e52", "Address": "1L2GM8eE7mJWLdo3HZS6su1832NX2txaac"},
    {"Bit Range": 24, "Private Key": "0000000000000000000000000000000000000000000000000000000000dc2a04", "Address": "1rSnXMr63jdCuegJFuidJqWxUPV7AtUf7"},
    {"Bit Range": 25, "Private Key": "0000000000000000000000000000000000000000000000000000000001fa5ee5", "Address": "15JhYXn6Mx3oF4Y7PcTAv2wVVAuCFFQNiP"},
    {"Bit Range": 26, "Private Key": "000000000000000000000000000000000000000000000000000000000340326e", "Address": "1JVnST957hGztonaWK6FougdtjxzHzRMMg"},
    {"Bit Range": 27, "Private Key": "0000000000000000000000000000000000000000000000000000000006ac3875", "Address": "128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k"},
    {"Bit Range": 28, "Private Key": "000000000000000000000000000000000000000000000000000000000d916ce8", "Address": "12jbtzBb54r97TCwW3G1gCFoumpckRAPdY"},
    {"Bit Range": 29, "Private Key": "0000000000000000000000000000000000000000000000000000000017e2551e", "Address": "19EEC52krRUK1RkUAEZmQdjTyHT7Gp1TYT"},
    {"Bit Range": 30, "Private Key": "000000000000000000000000000000000000000000000000000000003d94cd64", "Address": "1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps"},
    {"Bit Range": 31, "Private Key": "000000000000000000000000000000000000000000000000000000007d4fe747", "Address": "1LhE6sCTuGae42Axu1L1ZB7L96yi9irEBE"},
    {"Bit Range": 32, "Private Key": "00000000000000000000000000000000000000000000000000000000b862a62e", "Address": "1FRoHA9xewq7DjrZ1psWJVeTer8gHRqEvR"},
    {"Bit Range": 33, "Private Key": "00000000000000000000000000000000000000000000000000000001a96ca8d8", "Address": "187swFMjz1G54ycVU56B7jZFHFTNVQFDiu"},
    {"Bit Range": 34, "Private Key": "000000000000000000000000000000000000000000000000000000034a65911d", "Address": "1PWABE7oUahG2AFFQhhvViQovnCr4rEv7Q"},
    {"Bit Range": 35, "Private Key": "00000000000000000000000000000000000000000000000000000004aed21170", "Address": "1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb"},
    {"Bit Range": 36, "Private Key": "00000000000000000000000000000000000000000000000000000009de820a7c", "Address": "1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1"},
    {"Bit Range": 37, "Private Key": "0000000000000000000000000000000000000000000000000000001757756a93", "Address": "14iXhn8bGajVWegZHJ18vJLHhntcpL4dex"},
    {"Bit Range": 38, "Private Key": "00000000000000000000000000000000000000000000000000000022382facd0", "Address": "1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2"},
    {"Bit Range": 39, "Private Key": "0000000000000000000000000000000000000000000000000000004b5f8303e9", "Address": "122AJhKLEfkFBaGAd84pLp1kfE7xK3GdT8"},
    {"Bit Range": 40, "Private Key": "000000000000000000000000000000000000000000000000000000e9ae4933d6", "Address": "1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv"},
    {"Bit Range": 41, "Private Key": "00000000000000000000000000000000000000000000000000000153869acc5b", "Address": "1L5sU9qvJeuwQUdt4y1eiLmquFxKjtHr3E"},
    {"Bit Range": 42, "Private Key": "000000000000000000000000000000000000000000000000000002a221c58d8f", "Address": "1E32GPWgDyeyQac4aJxm9HVoLrrEYPnM4N"},
    {"Bit Range": 43, "Private Key": "000000000000000000000000000000000000000000000000000006bd3b27c591", "Address": "1PiFuqGpG8yGM5v6rNHWS3TjsG6awgEGA1"},
    {"Bit Range": 44, "Private Key": "00000000000000000000000000000000000000000000000000000e02b35a358f", "Address": "1CkR2uS7LmFwc3T2jV8C1BhWb5mQaoxedF"},
    {"Bit Range": 45, "Private Key": "0000000000000000000000000000000000000000000000000000122fca143c05", "Address": "1NtiLNGegHWE3Mp9g2JPkgx6wUg4TW7bbk"},
    {"Bit Range": 46, "Private Key": "00000000000000000000000000000000000000000000000000002ec18388d544", "Address": "1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP"},
    {"Bit Range": 47, "Private Key": "00000000000000000000000000000000000000000000000000006cd610b53cba", "Address": "1Pd8VvT49sHKsmqrQiP61RsVwmXCZ6ay7Z"},
    {"Bit Range": 48, "Private Key": "0000000000000000000000000000000000000000000000000000ade6d7ce3b9b", "Address": "1DFYhaB2J9q1LLZJWKTnscPWos9VBqDHzv"},
    {"Bit Range": 49, "Private Key": "000000000000000000000000000000000000000000000000000174176b015f4d", "Address": "12CiUhYVTTH33w3SPUBqcpMoqnApAV4WCF"},
    {"Bit Range": 50, "Private Key": "00000000000000000000000000000000000000000000000000022bd43c2e9354", "Address": "1MEzite4ReNuWaL5Ds17ePKt2dCxWEofwk"},
    {"Bit Range": 51, "Private Key": "00000000000000000000000000000000000000000000000000075070a1a009d4", "Address": "1NpnQyZ7x24ud82b7WiRNvPm6N8bqGQnaS"},
    {"Bit Range": 52, "Private Key": "000000000000000000000000000000000000000000000000000efae164cb9e3c", "Address": "15z9c9sVpu6fwNiK7dMAFgMYSK4GqsGZim"},
    {"Bit Range": 53, "Private Key": "00000000000000000000000000000000000000000000000000180788e47e326c", "Address": "15K1YKJMiJ4fpesTVUcByoz334rHmknxmT"},
    {"Bit Range": 54, "Private Key": "00000000000000000000000000000000000000000000000000236fb6d5ad1f43", "Address": "1KYUv7nSvXx4642TKeuC2SNdTk326uUpFy"},
    {"Bit Range": 55, "Private Key": "000000000000000000000000000000000000000000000000006abe1f9b67e114", "Address": "1LzhS3k3e9Ub8i2W1V8xQFdB8n2MYCHPCa"},
    {"Bit Range": 56, "Private Key": "000000000000000000000000000000000000000000000000009d18b63ac4ffdf", "Address": "17aPYR1m6pVAacXg1PTDDU7XafvK1dxvhi"},
    {"Bit Range": 57, "Private Key": "00000000000000000000000000000000000000000000000001eb25c90795d61c", "Address": "15c9mPGLku1HuW9LRtBf4jcHVpBUt8txKz"},
    {"Bit Range": 58, "Private Key": "00000000000000000000000000000000000000000000000002c675b852189a21", "Address": "1Dn8NF8qDyyfHMktmuoQLGyjWmZXgvosXf"},
    {"Bit Range": 59, "Private Key": "00000000000000000000000000000000000000000000000007496cbb87cab44f", "Address": "1HAX2n9Uruu9YDt4cqRgYcvtGvZj1rbUyt"},
    {"Bit Range": 60, "Private Key": "0000000000000000000000000000000000000000000000000fc07a1825367bbe", "Address": "1Kn5h2qpgw9mWE5jKpk8PP4qvvJ1QVy8su"},
    {"Bit Range": 61, "Private Key": "00000000000000000000000000000000000000000000000013c96a3742f64906", "Address": "1AVJKwzs9AskraJLGHAZPiaZcrpDr1U6AB"},
    {"Bit Range": 62, "Private Key": "000000000000000000000000000000000000000000000000363d541eb611abee", "Address": "1Me6EfpwZK5kQziBwBfvLiHjaPGxCKLoJi"},
    {"Bit Range": 63, "Private Key": "0000000000000000000000000000000000000000000000007cce5efdaccf6808", "Address": "1NpYjtLira16LfGbGwZJ5JbDPh3ai9bjf4"},
    {"Bit Range": 64, "Private Key": "000000000000000000000000000000000000000000000000f7051f27b09112d4", "Address": "16jY7qLJnxb7CHZyqBP8qca9d51gAjyXQN"},
    {"Bit Range": 65, "Private Key": "000000000000000000000000000000000000000000000001a838b13505b26867", "Address": "18ZMbwUFLMHoZBbfpCjUJQTCMCbktshgpe"},
    # {"Bit Range": 66, "Private Key": "000000000000000000000000000000000000000000000003ffffffffffffffff", "Address": "13zb1hqbwvsc2s7ztznp2g4undnnpdh5so"}
]

'''
# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Convert the Private Key to numeric format for model training
df['Private Key'] = df['Private Key'].apply(lambda x: int(x, 16))

# Step 2: Generate features and labels from the data
X = df[['Bit Range']]
y = df['Private Key']

# Step 3: Train a machine learning model using the features and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Use the model to predict potential private keys
y_pred = model.predict(X_test)

# Print the predictions
for bit_range, pred in zip(X_test['Bit Range'], y_pred):
    print(f"Bit Range: {bit_range}, Predicted Private Key: {pred:.0f}")

# Optional: Evaluate the model performance
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.0f}")

'''

df = pd.DataFrame(data)

# Convert Private Key from hex string to an integer for better handling
df['Private Key (int)'] = df['Private Key'].apply(lambda x: int(x, 16))

# Features and labels
X = df[['Bit Range']]
y = df['Private Key (int)']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=66)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the private key for Bit Range 66
bit_range_66 = np.array([[66]])
predicted_private_key_66 = model.predict(bit_range_66)

# Convert the predicted private key back to hex format
predicted_private_key_66_hex = hex(int(predicted_private_key_66[0]))[2:].zfill(64)

print(f"Predicted Private Key for Bit Range 66: {predicted_private_key_66_hex}")