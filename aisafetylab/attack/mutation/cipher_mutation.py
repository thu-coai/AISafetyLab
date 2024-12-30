from aisafetylab.attack.mutation.base_mutation import BaseMutation

class CaesarExpert(BaseMutation):
    def __init__(self, shift=3):
        super().__init__()
        self.shift = shift

    def encode(self, s):
        ans = ''
        for p in s:
            if 'a' <= p <= 'z':
                ans += chr(ord('a') + (ord(p) - ord('a') + self.shift) % 26)
            elif 'A' <= p <= 'Z':
                ans += chr(ord('A') + (ord(p) - ord('A') + self.shift) % 26)
            else:
                ans += p

        return ans

    def decode(self, s):
        ans = ''
        for p in s:
            if 'a' <= p <= 'z':
                ans += chr(ord('a') + (ord(p) - ord('a') - self.shift) % 26)
            elif 'A' <= p <= 'Z':
                ans += chr(ord('A') + (ord(p) - ord('A') - self.shift) % 26)
            else:
                ans += p
        return ans


class UnicodeExpert(BaseMutation):
    def __init__(self):
        super().__init__()

    def encode(self, s):
        ans = ''
        lines = s.split("\n")
        for line in lines:
            for c in line:
                byte_s = str(c.encode("unicode_escape"))
                if len(byte_s) > 8:
                    ans += byte_s[3:-1]
                else:
                    ans += byte_s[-2]
            ans += "\n"
        return ans

    def decode(self, s):
        ans = bytes(s, encoding="utf8").decode("unicode_escape")
        return ans


class BaseExpert(BaseMutation):
    def __init__(self):
        super().__init__()

    def encode(self, s):
        return s

    def decode(self, s):
        return s


class UTF8Expert(BaseMutation):
    def __init__(self):
        super().__init__()

    def encode(self, s):
        ans = ''

        lines = s.split("\n")
        for line in lines:
            for c in line:
                byte_s = str(c.encode("utf8"))
                if len(byte_s) > 8:
                    ans += byte_s[2:-1]
                else:
                    ans += byte_s[-2]
            ans += "\n"
        return ans

    def decode(self, s):
        ans = b''
        while len(s):
            if s.startswith("\\x"):
                ans += bytes.fromhex(s[2:4])
                s = s[4:]
            else:
                ans += bytes(s[0], encoding="utf8")
                s = s[1:]

        ans = ans.decode("utf8")
        return ans


class AsciiExpert(BaseMutation):
    def __init__(self):
        super().__init__()

    def encode(self, s):
        ans = ''

        lines = s.split("\n")
        for line in lines:
            for c in line:
                try:
                    ans += str(ord(c)) + " "
                except:
                    ans += c
            ans += "\n"
        return ans

    def decode(self, s):
        ans = ""
        lines = s.split("\n")
        for line in lines:
            cs = line.split()
            for c in cs:
                try:
                    ans += chr(int(c))
                except:
                    ans += c
        return ans

class GBKExpert(BaseMutation):
    def __init__(self):
        super().__init__()

    def encode(self, s):
        ans = ''

        lines = s.split("\n")
        for line in lines:
            for c in line:
                byte_s = str(c.encode("GBK"))
                if len(byte_s) > 8:
                    ans += byte_s[2:-1]
                else:
                    ans += byte_s[-2]
            ans += "\n"
        return ans

    def decode(self, s):
        ans = b''
        while len(s):
            if s.startswith("\\x"):
                ans += bytes.fromhex(s[2:4])
                s = s[4:]
            else:
                ans += bytes(s[0], encoding="GBK")
                s = s[1:]

        ans = ans.decode("GBK")
        return ans


class MorseExpert(BaseMutation):
    def __init__(self):
        super().__init__()

    def encode(self, s):
        s = s.upper()
        MORSE_CODE_DICT = {'A': '.-', 'B': '-...',
                           'C': '-.-.', 'D': '-..', 'E': '.',
                           'F': '..-.', 'G': '--.', 'H': '....',
                           'I': '..', 'J': '.---', 'K': '-.-',
                           'L': '.-..', 'M': '--', 'N': '-.',
                           'O': '---', 'P': '.--.', 'Q': '--.-',
                           'R': '.-.', 'S': '...', 'T': '-',
                           'U': '..-', 'V': '...-', 'W': '.--',
                           'X': '-..-', 'Y': '-.--', 'Z': '--..',
                           '1': '.----', '2': '..---', '3': '...--',
                           '4': '....-', '5': '.....', '6': '-....',
                           '7': '--...', '8': '---..', '9': '----.',
                           '0': '-----', ', ': '--..--', '.': '.-.-.-',
                           '?': '..--..', '/': '-..-.', '-': '-....-',
                           '(': '-.--.', ')': '-.--.-'}
        cipher = ''
        lines = s.split("\n")
        for line in lines:
            for letter in line:
                try:
                    if letter != ' ':
                        cipher += MORSE_CODE_DICT[letter] + ' '
                    else:
                        cipher += ' '
                except:
                    cipher += letter + ' '
            cipher += "\n"
        return cipher

    def decode(self, s):
        MORSE_CODE_DICT = {'A': '.-', 'B': '-...',
                           'C': '-.-.', 'D': '-..', 'E': '.',
                           'F': '..-.', 'G': '--.', 'H': '....',
                           'I': '..', 'J': '.---', 'K': '-.-',
                           'L': '.-..', 'M': '--', 'N': '-.',
                           'O': '---', 'P': '.--.', 'Q': '--.-',
                           'R': '.-.', 'S': '...', 'T': '-',
                           'U': '..-', 'V': '...-', 'W': '.--',
                           'X': '-..-', 'Y': '-.--', 'Z': '--..',
                           '1': '.----', '2': '..---', '3': '...--',
                           '4': '....-', '5': '.....', '6': '-....',
                           '7': '--...', '8': '---..', '9': '----.',
                           '0': '-----', ', ': '--..--', '.': '.-.-.-',
                           '?': '..--..', '/': '-..-.', '-': '-....-',
                           '(': '-.--.', ')': '-.--.-'}
        decipher = ''
        citext = ''
        lines = s.split("\n")
        for line in lines:
            for letter in line:
                while True and len(letter):
                    if letter[0] not in ['-', '.', ' ']:
                        decipher += letter[0]
                        letter = letter[1:]
                    else:
                        break
                try:
                    if (letter != ' '):
                        i = 0
                        citext += letter
                    else:
                        i += 1
                        if i == 2:
                            decipher += ' '
                        else:
                            decipher += list(MORSE_CODE_DICT.keys())[list(MORSE_CODE_DICT
                                                                          .values()).index(citext)]
                            citext = ''
                except:
                    decipher += letter
            decipher += '\n'
        return decipher



class AtbashExpert(BaseMutation):
    def __init__(self):
        super().__init__()

    def encode(self, text):
        ans = ''
        N = ord('z') + ord('a')
        for s in text:
            try:
                if s.isalpha():
                    ans += chr(N - ord(s))
                else:
                    ans += s
            except:
                ans += s
        return ans

    def decode(self, text):
        ans = ''
        N = ord('z') + ord('a')
        for s in text:
            try:
                if s.isalpha():
                    ans += chr(N - ord(s))
                else:
                    ans += s
            except:
                ans += s
        return ans


encode_expert_dict = {
    "baseline": BaseExpert(),
    "caesar": CaesarExpert(),
    "unicode": UnicodeExpert(),
    "morse": MorseExpert(),
    "atbash": AtbashExpert(),
    "utf": UTF8Expert(),
    "ascii": AsciiExpert(),
    "gbk": GBKExpert(),
    "selfcipher": BaseExpert()
}