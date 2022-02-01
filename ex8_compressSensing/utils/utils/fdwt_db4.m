function y=fdwt_db4(x)
qmf = MakeONFilter('Daubechies', 4);
y = FWT2_PO(real(x), 4, qmf) + 1i*FWT2_PO(imag(x), 4, qmf);
end