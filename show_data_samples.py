#!/usr/bin/env python3
"""
데이터 샘플 예쁘게 출력하는 스크립트
학습한 데이터들이 어떻게 생겼는지 확인할 수 있습니다.
"""
import os
import pickle
import numpy as np
import tiktoken
import random
import sys

# 색깔 초기화 (Windows 호환성)
try:
    from colorama import init, Fore, Back, Style
    init()
except ImportError:
    # colorama가 없으면 색깔 없이 출력
    class DummyFore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    Fore = DummyFore()
    Style = type('Style', (), {'BRIGHT': '', 'RESET_ALL': ''})()

def load_data_and_tokenizer():
    """데이터와 토크나이저 로드"""
    print(f"{Fore.CYAN}📁 데이터 로딩 중...{Style.RESET_ALL}")
    
    # 데이터 경로
    data_dir = 'data/Chat'
    train_file = os.path.join(data_dir, 'train4.bin')
    val_file = os.path.join(data_dir, 'val4.bin')
    meta_file = os.path.join(data_dir, 'meta.pkl')
    
    # 메타 정보 로드
    if not os.path.exists(meta_file):
        print(f"{Fore.RED}❌ meta.pkl 파일이 없습니다!{Style.RESET_ALL}")
        return None, None, None, None
    
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    
    vocab_size = meta['vocab_size']
    print(f"{Fore.GREEN}✓ Vocab size: {vocab_size:,}{Style.RESET_ALL}")
    
    # 토크나이저 로드
    if vocab_size == 200019:
        enc = tiktoken.get_encoding("o200k_base")
        tokenizer_name = "o200k_base"
    elif vocab_size in [50257, 50304]:
        enc = tiktoken.get_encoding("gpt2")
        tokenizer_name = "gpt2"
    else:
        enc = tiktoken.get_encoding("o200k_base")
        tokenizer_name = "o200k_base (fallback)"
    
    print(f"{Fore.GREEN}✓ 토크나이저: {tokenizer_name}{Style.RESET_ALL}")
    
    # 데이터 로드
    train_data = np.memmap(train_file, dtype=np.uint32, mode='r')
    val_data = np.memmap(val_file, dtype=np.uint32, mode='r')
    
    print(f"{Fore.GREEN}✓ 학습 데이터: {len(train_data):,} tokens{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ 검증 데이터: {len(val_data):,} tokens{Style.RESET_ALL}")
    
    return train_data, val_data, enc, vocab_size

def format_chat_text(text):
    """채팅 텍스트를 색깔로 예쁘게 포맷"""
    # 특수 태그들을 색깔로 바꾸기
    text = text.replace('<human>', f'{Fore.BLUE}{Style.BRIGHT}👤 Human:{Style.RESET_ALL}')
    text = text.replace('<bot>', f'{Fore.GREEN}{Style.BRIGHT}🤖 Bot:{Style.RESET_ALL}')
    text = text.replace('<endOfText>', f'{Fore.YELLOW}[END]{Style.RESET_ALL}')
    text = text.replace('<endOftext>', f'{Fore.YELLOW}[END]{Style.RESET_ALL}')
    text = text.replace('<|endoftext|>', f'{Fore.YELLOW}[END]{Style.RESET_ALL}')
    
    return text

def show_sample_conversations(train_data, enc, num_samples=5, tokens_per_sample=512):
    """샘플 대화들을 예쁘게 출력"""
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"🗣️  샘플 대화들 ({num_samples}개)")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    for i in range(num_samples):
        # 랜덤 위치에서 샘플 추출
        start_idx = random.randint(0, len(train_data) - tokens_per_sample)
        sample_tokens = train_data[start_idx:start_idx + tokens_per_sample]
        
        try:
            # 토큰을 텍스트로 디코딩
            sample_text = enc.decode(sample_tokens.tolist())
            
            print(f"\n{Fore.CYAN}{Style.BRIGHT}📝 샘플 #{i+1} (토큰 {start_idx:,} ~ {start_idx + tokens_per_sample:,}){Style.RESET_ALL}")
            print(f"{Fore.WHITE}{'─'*50}{Style.RESET_ALL}")
            
            # 채팅 형식으로 포맷
            formatted_text = format_chat_text(sample_text)
            
            # 너무 긴 텍스트는 자르기
            lines = formatted_text.split('\n')
            for line in lines[:20]:  # 최대 20줄만 출력
                if line.strip():
                    print(line.strip())
            
            if len(lines) > 20:
                print(f"{Fore.YELLOW}... (총 {len(lines)}줄 중 20줄만 표시){Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ 디코딩 실패: {e}{Style.RESET_ALL}")

def show_data_statistics(train_data, val_data, enc, vocab_size):
    """데이터 통계 보여주기"""
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"📊 데이터 통계")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    # 기본 통계
    total_tokens = len(train_data) + len(val_data)
    print(f"{Fore.CYAN}📈 전체 토큰 수: {total_tokens:,}")
    print(f"   - 학습용: {len(train_data):,} ({len(train_data)/total_tokens*100:.1f}%)")
    print(f"   - 검증용: {len(val_data):,} ({len(val_data)/total_tokens*100:.1f}%){Style.RESET_ALL}")
    
    # 토큰 범위 확인
    train_min, train_max = train_data.min(), train_data.max()
    val_min, val_max = val_data.min(), val_data.max()
    
    print(f"\n{Fore.YELLOW}🔢 토큰 값 범위:")
    print(f"   - 학습 데이터: {train_min} ~ {train_max}")
    print(f"   - 검증 데이터: {val_min} ~ {val_max}")
    print(f"   - 어휘 크기: {vocab_size:,}{Style.RESET_ALL}")
    
    # 범위 체크
    if train_max >= vocab_size or val_max >= vocab_size:
        print(f"{Fore.RED}⚠️  경고: 일부 토큰이 어휘 크기를 초과합니다!{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}✓ 모든 토큰이 어휘 범위 내에 있습니다.{Style.RESET_ALL}")
    
    # 샘플 크기 정보
    approx_words = total_tokens * 0.75  # 대략적인 단어 수 (토큰 대 단어 비율)
    print(f"\n{Fore.CYAN}📝 대략적인 크기:")
    print(f"   - 예상 단어 수: {approx_words:,.0f}")
    print(f"   - 예상 페이지 수: {approx_words/250:,.0f} (250단어/페이지 기준){Style.RESET_ALL}")

def show_tokenizer_examples(enc):
    """토크나이저 동작 예시"""
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"🔤 토크나이저 동작 예시")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    examples = [
        "Hello world!",
        "<human>안녕하세요<endOfText><bot>안녕하세요! 반갑습니다.<endOfText>",
        "The quick brown fox jumps over the lazy dog.",
        "Python programming is fun! 🐍",
        "<|endoftext|>",
    ]
    
    for text in examples:
        try:
            tokens = enc.encode(text, allowed_special={"<|endoftext|>"}, disallowed_special=())
            decoded = enc.decode(tokens)
            
            print(f"\n{Fore.WHITE}원본: {Style.RESET_ALL}{text}")
            print(f"{Fore.YELLOW}토큰 수: {len(tokens)}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}토큰들: {tokens[:10]}{'...' if len(tokens) > 10 else ''}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}디코딩: {Style.RESET_ALL}{decoded}")
            print(f"{Fore.WHITE}{'─'*40}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ 토큰화 실패 ({text}): {e}{Style.RESET_ALL}")

def main():
    print(f"{Fore.MAGENTA}{Style.BRIGHT}")
    print("=" * 60)
    print("🎯 nanoChatGPT 데이터 분석기")
    print("=" * 60)
    print(f"{Style.RESET_ALL}")
    
    # 데이터 로드
    train_data, val_data, enc, vocab_size = load_data_and_tokenizer()
    if train_data is None:
        return
    
    # 메뉴
    while True:
        print(f"\n{Fore.CYAN}📋 메뉴를 선택하세요:")
        print(f"1. 📊 데이터 통계 보기")
        print(f"2. 🗣️  샘플 대화 보기")
        print(f"3. 🔤 토크나이저 예시 보기")
        print(f"4. 🔍 전체 분석 (모든 항목)")
        print(f"5. 🚪 종료{Style.RESET_ALL}")
        
        try:
            choice = input(f"\n{Fore.WHITE}선택 (1-5): {Style.RESET_ALL}").strip()
            
            if choice == '1':
                show_data_statistics(train_data, val_data, enc, vocab_size)
            elif choice == '2':
                num_samples = input(f"{Fore.WHITE}샘플 수 (기본값 5): {Style.RESET_ALL}").strip()
                num_samples = int(num_samples) if num_samples.isdigit() else 5
                show_sample_conversations(train_data, enc, num_samples)
            elif choice == '3':
                show_tokenizer_examples(enc)
            elif choice == '4':
                show_data_statistics(train_data, val_data, enc, vocab_size)
                show_sample_conversations(train_data, enc, 3)
                show_tokenizer_examples(enc)
            elif choice == '5':
                print(f"{Fore.GREEN}👋 안녕히 가세요!{Style.RESET_ALL}")
                break
            else:
                print(f"{Fore.RED}❌ 잘못된 선택입니다.{Style.RESET_ALL}")
                
        except KeyboardInterrupt:
            print(f"\n{Fore.GREEN}👋 안녕히 가세요!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}❌ 오류 발생: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
