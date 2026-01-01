def get_scores():
    """
    讓使用者輸入成績，直到輸入 q 為止
    回傳一個成績清單
    """
    scores = []

    while True:
        user_input = input("請輸入成績（輸入 q 結束）：")

        if user_input.lower() == "q":
            break

        try:
            score = float(user_input)
            if 0 <= score <= 100:
                scores.append(score)
            else:
                print("成績請輸入 0～100 之間")
        except ValueError:
            print("請輸入有效的數字")

    return scores


def analyze_scores(scores):
    """
    分析成績：平均、最高、最低
    """
    average = sum(scores) / len(scores)
    highest = max(scores)
    lowest = min(scores)

    return average, highest, lowest


def main():
    print("📘 成績分析系統")

    scores = get_scores()

    if not scores:
        print("沒有輸入任何成績")
        return

    average, highest, lowest = analyze_scores(scores)

    print("\n📊 分析結果")
    print(f"平均分數：{average:.2f}")
    print(f"最高分數：{highest}")
    print(f"最低分數：{lowest}")


if __name__ == "__main__":
    main()
