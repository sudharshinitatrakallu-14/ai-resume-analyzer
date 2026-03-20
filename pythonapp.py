from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("===== AI Resume Analyzer =====\n")

resume = input("Paste your resume text:\n")
job_desc = input("\nPaste job description:\n")

text = [resume, job_desc]

cv = CountVectorizer()
count_matrix = cv.fit_transform(text)

similarity = cosine_similarity(count_matrix)[0][1]

score = round(similarity * 100, 2)

print("\n🔹 Resume Match Score:", score, "%")

if score < 40:
    print("❌ Low match. Add more relevant skills and keywords.")
elif score < 70:
    print("⚠️ متوسط match. Improve by adding more project-related keywords.")
else:
    print("✅ Great match! Your resume fits well.")

print("\n💡 Suggestions:")
print("- Add more relevant technical skills")
print("- Include project keywords")
print("- Tailor resume based on job description")