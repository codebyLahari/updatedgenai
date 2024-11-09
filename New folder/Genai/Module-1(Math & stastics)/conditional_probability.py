def calculate_probability():
# Total number of cards
    total_cards = 52
    
    # Number of black cards p(B) 
    total_black_cards = 26
    
    # Number of black kings p(A)
    total_black_kings = 2
    
    # Probability of drawing a black king, P(A)
    P_A = total_black_kings / total_cards  # P(A) = P(Black King)
    
    # Probability of drawing a black card, P(B)
    P_B = total_black_cards / total_cards  # P(B) = P(Black Card)
    
    # Probability of drawing a black card given that a black king is drawn, P(B | A)
    P_B_given_A = 1  # If you have drawn a black king, it is definitely a black card.
    
    # Applying Bayes' Theorem to find P(A | B)
    P_A_given_B = (P_B_given_A * P_A) / P_B
    
    return P_A_given_B

# Main function
if __name__ == "__main__":
    print("Calculating the probability of drawing a black king given that a black card is drawn...")
    probability = calculate_probability()
    print(f"P(Black King | Black Card) = {probability:.2f} or {probability * 100:.2f}%")
