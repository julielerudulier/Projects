def reverse_words(text):
    words = text.split()
    reversed_sentence = ' '.join(reversed(words))
    return reversed_sentence

text = input("Please write a sentence: ")
reversed_text = reverse_words(text)

print(f"The initial sentence was: {text}")
print(f"The reversed sentence is: {reversed_text}")
