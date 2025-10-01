"""
Quick test script - Nhanh chóng test model đã train
"""
import os
from inference import QuestionGenerator


def main():
    print("=" * 70)
    print("⚡ QUICK TEST - QUESTION GENERATION")
    print("=" * 70)
    
    # Check if model exists
    if not os.path.exists('checkpoints/best_model.pt'):
        print("\n❌ Chưa có model đã train!")
        print("\n📋 Các bước:")
        print("1. Chạy: python Train.py")
        print("2. Đợi training hoàn thành (~1-2 giờ)")
        print("3. Chạy lại script này")
        return
    
    # Load model
    print("\n🔄 Loading model...")
    generator = QuestionGenerator()
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("💬 INTERACTIVE MODE")
    print("=" * 70)
    print("Nhập context để tạo câu hỏi (hoặc 'quit' để thoát)")
    print("-" * 70)
    
    while True:
        print("\n📝 Context:")
        context = input("> ").strip()
        
        if context.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not context:
            print("⚠️  Context không được rỗng!")
            continue
        
        # Generate question
        print("\n🤔 Generating question...")
        question = generator.generate(context, method='greedy', max_len=50)
        
        print(f"\n✅ Generated Question:")
        print(f"   {question}")
        
        # Generate alternative (more diverse)
        question_alt = generator.generate(context, method='sampling', temperature=0.8, max_len=50)
        print(f"\n🎲 Alternative (Sampling):")
        print(f"   {question_alt}")
        
        print("-" * 70)


if __name__ == "__main__":
    main()

