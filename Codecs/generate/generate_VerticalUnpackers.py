#!/usr/bin/env python

import generate_ScalarUnpackers

class Base(object):
	def __init__(self, name, kGroupSize):
		self.name = name
		self.kGroupSize = kGroupSize
		
	def datatype(self):
		pass

	def set1(self):
		pass

	def loadu(self):
		pass

	def storeu(self):
		pass

	def AND(self):
		pass

	def OR(self):
		pass

	def srli(self):
		pass

	def slli(self):
		pass

class SSE(Base):
	def __init__(self):
		Base.__init__(self, "SSE", 4)
	
	def datatype(self):
		return "__m128i"

	def set1(self):
		return "_mm_set1_epi32"

	def loadu(self):
		return "_mm_loadu_si128"

	def storeu(self):
		return "_mm_storeu_si128"

	def AND(self):
		return "_mm_and_si128"

	def OR(self):
		return "_mm_or_si128"

	def srli(self):
		return "_mm_srli_epi32"

	def slli(self):
		return "_mm_slli_epi32"

class AVX(Base):
	def __init__(self):
		Base.__init__(self, "AVX", 8)
	
	def datatype(self):
		return "__m256i"

	def set1(self):
		return "_mm256_set1_epi32"

	def loadu(self):
		return "_mm256_loadu_si256"

	def storeu(self):
		return "_mm256_storeu_si256"

	def AND(self):
		return "_mm256_and_si256"

	def OR(self):
		return "_mm256_or_si256"

	def srli(self):
		return "_mm256_srli_epi32"

	def slli(self):
		return "_mm256_slli_epi32"


def unpackerBodyIMP(b, ins, packSize, IsRiceCoding):
    DataType = ins.datatype() # __m128i or __m256i
    InReg = "c%d_load_rslt%s" %(b, DataType[1:])   # input register
    OutReg = "c%d_rslt%s" %(b, DataType[1:])       # output register
    Set1 = ins.set1()      # set1 instruction
    Loadu = ins.loadu()    # load instruction
    Storeu = ins.storeu()  # store instruction
    And = ins.AND()        # and instruction
    Or = ins.OR()          # or instuction
    Srli = ins.srli()      # srli instruction
    Slli = ins.slli()      # slli instruction

    AndMask = "SIMDMasks::%s2_and_msk_%s[%d]" % (ins.name, DataType[2:], b)  # and mask

    lines = []
    lines.append("\n\t\t%s %s, %s;\n" % (DataType, InReg, OutReg)) # input & output register definito
    for i in range(0, packSize / ins.kGroupSize):
        idx = (i * b) >> 5      # input index
        shift = (i * b) & 0x1f  # bits used (mod 32)

        if b < 32 - shift:
            if shift == 0:
                lines.append("\n\n\t\t%s = %s(in + %d);\n" % (InReg, Loadu, idx))        # load current input
                lines.append("\n\t\t%s = %s( %s, %s );" % (OutReg, And, InReg, AndMask)) # mask
            else:
                lines.append("\n\t\t%s = %s( %s(%s, %d), %s );" % (OutReg, And, Srli, InReg, shift, AndMask)) # shift right & mask
        elif b == 32 - shift:
            lines.append("\n\t\t%s = %s(%s, %d);" % (OutReg, Srli, InReg, shift))   # shift right
        else: # codewords spanning across consecutive groups
            lines.append("\n\n\t\t%s = %s(%s, %d);" % (OutReg, Srli, InReg, shift)) # shift right
            lines.append("\n\t\t%s = %s(in + %d);\n" % (InReg, Loadu, idx + 1))     # load next input
            lines.append("\n\t\t%s = %s( %s(%s, %s(%s, 32 - %d)), %s );" % (OutReg, And, Or, OutReg, Slli, InReg, shift, AndMask)) # concatenate

        if IsRiceCoding:
            lines.append("\n\t\t%s = %s( %s, %s(%s(quotient + %d), %d) );" %(OutReg, Or, OutReg, Slli, Loadu, i, b))
        lines.append("\n\t\t%s(out + %d, %s);\n" % (Storeu, i, OutReg)) # store 
    return "".join(lines)


def unpackerBody(b, ins, packSize):
    lines = []
    if b == 0:
        lines.append("\n\tif (!IsRiceCoding) { // For NewPFor and OptPFor.");
        lines.append("\n\t\tuint32_t *outPtr = reinterpret_cast<uint32_t *>(out);")
        lines.append("\n\t\tfor (uint32_t valuesUnpacked = 0; valuesUnpacked < %d; valuesUnpacked += 64) {" % packSize)
        lines.append("\n\t\t\tmemset64(outPtr);")
        lines.append("\n\t\t\toutPtr += 64;")
        lines.append("\n\t\t}")
        lines.append("\n\t}")
        lines.append("\n\telse { // For Rice and OptRice.")
        lines.append("\n\t\tuint32_t *outPtr = reinterpret_cast<uint32_t *>(out);")
        lines.append("\n\t\tconst uint32_t *quoPtr = reinterpret_cast<const uint32_t *>(quotient);")
        lines.append("\n\t\tfor (uint32_t valuesUnpacked = 0; valuesUnpacked < %d; valuesUnpacked += 32) {" % packSize)
        lines.append("\n\t\t\tmemcpy32(outPtr, quoPtr);")
        lines.append("\n\t\t\toutPtr += 32;")
        lines.append("\n\t\t\tquoPtr += 32;")
        lines.append("\n\t\t}")
        lines.append("\n\t}\n")
        return "".join(lines)

    if b == 32:
        lines.append("\n\tuint32_t *outPtr = reinterpret_cast<uint32_t *>(out);")
        lines.append("\n\tconst uint32_t *inPtr = reinterpret_cast<const uint32_t *>(in);")
        lines.append("\n\tfor (uint32_t valuesUnpacked = 0; valuesUnpacked < %d; valuesUnpacked += 32) {" % packSize)
        lines.append("\n\t\tmemcpy32(outPtr, inPtr);")
        lines.append("\n\t\toutPtr += 32;")
        lines.append("\n\t\tinPtr += 32;")
        lines.append("\n\t}\n")
        return "".join(lines)

    lines.append("\n\tif (!IsRiceCoding) { // For NewPFor and OptPFor.")
    lines.append(unpackerBodyIMP(b, ins, packSize, False))
    lines.append("\t}")
    lines.append("\n\telse { // For Rice and OptRice.")
    lines.append(unpackerBodyIMP(b, ins, packSize, True))
    lines.append("\t}\n")
    return "".join(lines)


def packerBody(b, ins, packSize, withoutmask):
    lines = []
    if b == 32:
        lines.append("\n\tuint32_t *outPtr = reinterpret_cast<uint32_t *>(out);")
        lines.append("\n\tconst uint32_t *inPtr = reinterpret_cast<const uint32_t *>(in);")
        lines.append("\n\tfor (uint32_t valuesPacked = 0; valuesPacked < %d; valuesPacked += 32) {" % packSize)
        lines.append("\n\t\tmemcpy32(outPtr, inPtr);")
        lines.append("\n\t\toutPtr += 32;")
        lines.append("\n\t\tinPtr += 32;")
        lines.append("\n\t}\n")
        return "".join(lines)

    DataType = ins.datatype() # __m128i or __m256i
    InReg = "c%d_load_rslt%s" %(b, DataType[1:])   # input register
    OutReg = "c%d_rslt%s" %(b, DataType[1:])       # output register
    Set1 = ins.set1()      # set1 instruction
    Loadu = ins.loadu()    # load instruction
    Storeu = ins.storeu()  # store instruction
    And = ins.AND()        # and instruction
    Or = ins.OR()          # or instuction
    Srli = ins.srli()      # srli instruction
    Slli = ins.slli()      # slli instruction

    AndMask = "SIMDMasks::%s2_and_msk_%s[%d]" % (ins.name, DataType[2:], b)  # and mask

    lines.append("\n\t%s %s, %s;\n" % (DataType, InReg, OutReg)) # input & output register definition

    for i in range(0, packSize / ins.kGroupSize):
        idx = (i * b) >> 5      # output index
        shift = (i * b) & 0x1f  # bits used (mod 32)

        if withoutmask:
            lines.append("\n\n\t%s = %s(in + %d);" %(InReg, Loadu, i)) # load
        else:
            lines.append("\n\n\t%s = %s(%s(in + %d), %s);" %(InReg, And, Loadu, i, AndMask)) # mask and load

        if shift == 0:
            lines.append("\n\t%s = %s;" % (OutReg, InReg)) # assign
        else:
            lines.append("\n\t%s = %s(%s, %s(%s, %d));" % (OutReg, Or, OutReg, Slli, InReg, shift)) # concatenate and assign

        if b == 32 - shift:
            lines.append("\n\n\t%s(out + %d, %s);\n" % (Storeu, idx, OutReg)) # store
        elif b > 32 - shift:
            lines.append("\n\n\t%s(out + %d, %s);\n" % (Storeu, idx, OutReg)) # store
            lines.append("\n\n\t%s = %s(%s, 32 - %d);" %(OutReg, Srli, InReg, shift)) # shift right

    return "".join(lines)

def generate(codecName, className, ins, packSize):
    fileName = "../include/%s.h" % codecName
    target = open(fileName, 'w')

    lines = []
    lines.append("#ifndef CODECS_%s_H_" % codecName.upper())
    lines.append("\n#define CODECS_%s_H_\n" % codecName.upper())
    for b in range(0, 33):
        lines.append("\n// %d-bit" % b)  # comment
        lines.append("\ntemplate <bool IsRiceCoding>")
        lines.append("\nvoid %s<%s, IsRiceCoding>::verticalunpack_c%d(%s *  __restrict__  out,\n\t\tconst %s *  __restrict__  in) {" % (className, ins.name, b, ins.datatype(), ins.datatype())) # prototype
        lines.append(unpackerBody(b, ins, packSize)) # function body
        lines.append("}\n") # closing brace
	
    for b in range(1, 33):
        lines.append("\n// %d-bit" % b)
        lines.append("\ntemplate <bool IsRiceCoding>")
        lines.append("\nvoid %s<%s, IsRiceCoding>::verticalpackwithoutmask_c%d(%s *  __restrict__  out,\n\t\tconst %s *  __restrict__  in) {" % (className, ins.name, b, ins.datatype(), ins.datatype()))
        lines.append(packerBody(b, ins, packSize, True))
        lines.append("}\n")

    for b in range(1, 33):
        lines.append("\n// %d-bit" % b)
        lines.append("\ntemplate <bool IsRiceCoding>")
        lines.append("\nvoid %s<%s, IsRiceCoding>::verticalpack_c%d(%s *  __restrict__  out,\n\t\tconst %s *  __restrict__  in) {" % (className, ins.name, b, ins.datatype(), ins.datatype()))
        lines.append(packerBody(b, ins, packSize, False))
        lines.append("}\n")

    lines.append("\n#endif /* CODECS_%s_H_ */\n" % fileName.upper())
    target.write("".join(lines))

    target.close()


def main():
	generate_ScalarUnpackers.generate("VerScalarUnpacker", "VerUnpacker", 128, 4) # VerScalarUnpackerIMP.h

	generate("VerSSEUnpacker", "VerUnpacker", SSE(), 128) # VerSSEUnpackerIMP.h

	generate("VerAVXUnpacker", "VerUnpacker", AVX(), 256) # VerAVXUnpackerIMP.h

if __name__ == "__main__":
	main()
