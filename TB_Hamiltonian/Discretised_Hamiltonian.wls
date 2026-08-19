#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[NotebookDirectory[]];
e = 1.602*10^-19; (*C*)
hbar=1.05457*10^-34; (*J.s*)
m0=me=9.109*10^-31; (*kg*)
a0 = 5.292*10^-11; (*m*)


(*"""Optical transition matrix elements"""*)
Pperp=0.4829*hbar/(a0*e * 10^-10); (*eV . A*)
Ppar=0.6431*hbar/(a0*e * 10^-10);(*eV . A*)

(*eV . A = J.m/(e * 10^-10)*)
(*eV . A^2 = J.m/(e * 10^-20)*)

(*"""conduction band effective parameters"""*)
Ac1par=hbar^2/(2*m0 * e * 10^-20); (*eV . A^2*)
Ac2perp=4.1565*hbar^2/(2*m0 * e * 10^-20); (*eV . A^2*)
Ac1perp=9.5120*hbar^2/(2*m0 * e * 10^-20); (*eV . A^2*)
Ac2par=2.4091*hbar^2/(2*m0 * e * 10^-20); (*eV . A^2*)


(*"""valance band effective parameters"""*)
A1=-4.3636*hbar^2/(2*m0 * e * 10^-20); (*eV . A^2*)
A2=-2.0833*hbar^2/(2*m0 * e * 10^-20); (*eV . A^2*)
A3=2.4545*hbar^2/(2*m0 * e * 10^-20); (*eV . A^2*)
A4=-2.7504*hbar^2/(2*m0 * e * 10^-20); (*eV . A^2*)
A5=-2.7232*hbar^2/(2*m0 * e * 10^-20); (*eV . A^2*)
A6=-3.5421*hbar^2/(2*m0 * e * 10^-20); (*eV . A^2*)


(*"""Energy splittings"""*)
Delta1=0.2688 ; (*eV*)
Delta2=0.0934; (*eV*) 
Delta3=0.0908; (*eV*)

Ev=-0.3622; (*eV*)
Ecbp1=0.632 ; (*eV*)
Ecb=0.298; (*eV*)

H[kx_,ky_,kz_]:=Module[{kplus,kminus,kin,C1,C2,alpha,beta,Htmp},
(*momentum combinations*)
kplus=kx+I ky;
kminus=kx-I ky;
(*kinetic term*)
kin= 0;
(*conduction band parameters*)
C1=Ac1perp (kx^2+ky^2)+Ac1par kz^2;
C2=Ac2perp (kx^2+ky^2)+Ac2par kz^2;
(*valence band terms*)
beta=A1 kz^2+A2 (kx^2+ky^2);
alpha=A3 kz^2+A4 (kx^2+ky^2);
(*initialize Hamiltonian*)
Htmp=ConstantArray[0,{10,10}];
(*Fill nonzero entries*)
Htmp[[1,1]]=kin+C2+Ecbp1;
Htmp[[1,3]]=-(Sqrt[2] hbar kplus Pperp)/(2 m0);
Htmp[[1,4]]=(Sqrt[2] hbar kminus Pperp)/(2 m0);
Htmp[[1,5]]=(hbar kz Ppar)/m0;
Htmp[[2,2]]=kin+C1+Ecb;
Htmp[[3,1]]=-(Sqrt[2] hbar kminus Pperp)/(2 m0);
Htmp[[3,3]]=kin+Delta1+Delta2+Ev+alpha+beta;
Htmp[[3,4]]=-A5 (kminus)^2;
Htmp[[3,5]]=-A6 kminus kz;
Htmp[[4,1]]=(Sqrt[2] hbar kplus Pperp)/(2 m0);
Htmp[[4,3]]=-A5 (kplus)^2;
Htmp[[4,4]]=kin+Delta1-Delta2+Ev+alpha+beta;
Htmp[[4,5]]=A6 kplus kz;
Htmp[[4,10]]=Sqrt[2] Delta3;
Htmp[[5,1]]=(hbar kz Ppar)/m0;
Htmp[[5,3]]=-A6 kplus kz;
Htmp[[5,4]]=A6 kminus kz;
Htmp[[5,5]]=kin+Ev+beta;
Htmp[[5,9]]=Sqrt[2] Delta3;
Htmp[[6,6]]=kin+C2+Ecbp1;
Htmp[[6,8]]=(Sqrt[2] hbar kminus Pperp)/(2 m0);
Htmp[[6,9]]=-(Sqrt[2] hbar kplus Pperp)/(2 m0);
Htmp[[6,10]]=(hbar kz Ppar)/m0;
Htmp[[7,7]]=kin+C1+Ecb;
Htmp[[8,6]]=(Sqrt[2] hbar kplus Pperp)/(2 m0);
Htmp[[8,8]]=kin+Delta1+Delta2+Ev+alpha+beta;
Htmp[[8,9]]=-A5 (kplus)^2;
Htmp[[8,10]]=A6 kplus kz;
Htmp[[9,5]]=Sqrt[2] Delta3;
Htmp[[9,6]]=-(Sqrt[2] hbar kminus Pperp)/(2 m0);
Htmp[[9,8]]=-A5 (kminus^2);
Htmp[[9,9]]=kin+Delta1-Delta2+Ev+alpha+beta;
Htmp[[9,10]]=-A6 kminus kz;
Htmp[[10,4]]=Sqrt[2] Delta3;
Htmp[[10,6]]=(hbar kz Ppar)/m0;
Htmp[[10,8]]=A6 kminus kz;
Htmp[[10,9]]=-A6 kplus kz;
Htmp[[10,10]]=kin+Ev+beta;
(*return Hamiltonian matrix*)Htmp]

Evals = Sort[Eigenvalues[H[0.0,0,0]]];
Print["Energy at Gamma point: "]
Print["Evals= ",Evals]


nPoints=400;
kpath = N[Subdivide[0,0.1,nPoints]];
band1 = Transpose[Table[Sort[Eigenvalues[H[i,0,0]]],{i,kpath}]];
banddata = Table[Table[{kpath[[i]],band1[[j,i]]},{i,Length[kpath]}],{j,10}];
bandplot1 = ListLinePlot[banddata,Frame->True,PlotStyle->Red,FrameLabel->{"\!\(\*SubscriptBox[\(k\), \(x\)]\) (1/\[Angstrom])","Energy (eV)"},FrameStyle->Directive[Black],
LabelStyle->{FontFamily->"Times New Roman",15},ImageSize->250,AspectRatio->1.5,PlotRange->{-1,1.5}]


(*Repalce kx = Sin kx*)
(*NN = 20;
L = 300;
a = L / (NN+1);*)
Print[N[a]]
Hdis[a_,kx_,ky_,kz_]:=Module[{kplus,kminus,kin,C1,C2,alpha,beta,Htmp},
(*momentum combinations*)
kplus=(1/a*Sin[kx*a ]+I 1/a*Sin[ky*a ]);
kminus=(1/a*Sin[kx*a ]-I 1/a*Sin[ky*a ]);

(*kinetic term*)
kin=0;(*hbar^2 (2/(a^2)*(1-Cos[kx*a ])+2/(a^2)*(1-Cos[ky*a ])+2/(a^2)*(1-Cos[kz*a ]))/(2 m0* e * 10^-20);*)

(*conduction band parameters*)
C1=Ac1perp (2/(a^2)*(1-Cos[kx*a ])+2/(a^2)*(1-Cos[ky*a ]))+Ac1par 2/(a^2)*(1-Cos[kz*a ]);
C2=Ac2perp (2/(a^2)*(1-Cos[kx*a ])+2/(a^2)*(1-Cos[ky*a ]))+Ac2par 2/(a^2)*(1-Cos[kz*a ]);

(*valence band terms*)
beta=A1 2/(a^2)*(1-Cos[kz*a ])+A2 (2/(a^2)*(1-Cos[kx*a ])+2/(a^2)*(1-Cos[ky*a ]));
alpha=A3 2/(a^2)*(1-Cos[kz*a ])+A4 (2/(a^2)*(1-Cos[kx*a ])+2/(a^2)*(1-Cos[ky*a ]));

(*initialize Hamiltonian*)
Htmp=ConstantArray[0,{10,10}];

(*Fill nonzero entries*)
Htmp[[1,1]]=kin+C2+Ecbp1;
Htmp[[1,3]]=-(Sqrt[2] hbar kplus Pperp)/(2 m0);
Htmp[[1,4]]=(Sqrt[2] hbar kminus Pperp)/(2 m0);
Htmp[[1,5]]=(hbar 1/a*Sin[kz*a] Ppar)/m0;

Htmp[[2,2]]=kin+C1+Ecb;

Htmp[[3,1]]=-(Sqrt[2] hbar kminus Pperp)/(2 m0);
Htmp[[3,3]]=kin+Delta1+Delta2+Ev+alpha+beta;
Htmp[[3,4]]=-A5 * -2/a^2 * ( Cos[kx a] - Cos[ky a] + I Sin[kx a] Sin[ky a]);
Htmp[[3,5]]=-A6 kminus 1/a*Sin[kz*a];

Htmp[[4,1]]=(Sqrt[2] hbar kplus Pperp)/(2 m0);
Htmp[[4,3]]=-A5 *-2/a^2 * ( Cos[kx a] - Cos[ky a] - I Sin[kx a] Sin[ky a]);
Htmp[[4,4]]=kin+Delta1-Delta2+Ev+alpha+beta;
Htmp[[4,5]]=A6 kplus 1/a*Sin[kz*a];
Htmp[[4,10]]=Sqrt[2] Delta3;

Htmp[[5,1]]=(hbar 1/a*Sin[kz*a] Ppar)/m0;
Htmp[[5,3]]=-A6 kplus 1/a*Sin[kz*a];
Htmp[[5,4]]=A6 kminus 1/a*Sin[kz*a];
Htmp[[5,5]]=kin+Ev+beta;
Htmp[[5,9]]=Sqrt[2] Delta3;

Htmp[[6,6]]=kin+C2+Ecbp1;
Htmp[[6,8]]=(Sqrt[2] hbar kminus Pperp)/(2 m0);
Htmp[[6,9]]=-(Sqrt[2] hbar kplus Pperp)/(2 m0);
Htmp[[6,10]]=(hbar 1/a*Sin[kz*a] Ppar)/m0;

Htmp[[7,7]]=kin+C1+Ecb;

Htmp[[8,6]]=(Sqrt[2] hbar kplus Pperp)/(2 m0);
Htmp[[8,8]]=kin+Delta1+Delta2+Ev+alpha+beta;
Htmp[[8,9]]=-A5 * -2/a^2 * ( Cos[kx a] - Cos[ky a] - I Sin[kx a] Sin[ky a]);
Htmp[[8,10]]=A6 kplus 1/a*Sin[kz*a];

Htmp[[9,5]]=Sqrt[2] Delta3;
Htmp[[9,6]]=-(Sqrt[2] hbar kminus Pperp)/(2 m0);
Htmp[[9,8]]=-A5 *-2/a^2 * ( Cos[kx a] - Cos[ky a] + I Sin[kx a] Sin[ky a]);
Htmp[[9,9]]=kin+Delta1-Delta2+Ev+alpha+beta;
Htmp[[9,10]]=-A6 kminus 1/a*Sin[kz*a];

Htmp[[10,4]]=Sqrt[2] Delta3;
Htmp[[10,6]]=(hbar 1/a*Sin[kz*a] Ppar)/m0;
Htmp[[10,8]]=A6 kminus 1/a*Sin[kz*a];
Htmp[[10,9]]=-A6 kplus 1/a*Sin[kz*a];
Htmp[[10,10]]=kin+Ev+beta;

Htmp]

NN = 10;
L = 300;

Evals = Sort[Eigenvalues[Hdis[L/(NN + 1),0,0,0]]];
Print["Energy at Gamma point: "]
Print["Evals= ",Evals]
(*FullSimplify[Hdis[kx,ky,kz]]*)
(*FullSimplify[Hdis[0,0,0]]//MatrixForm*)


nPoints=400;
kpath = N[Subdivide[0,0.1,nPoints]];
a = L/(NN + 1);
band1 = Transpose[Table[Sort[Eigenvalues[Hdis[a,i,0,0]]],{i,kpath}]];
banddata = Table[Table[{kpath[[i]],band1[[j,i]]},{i,Length[kpath]}],{j,10}];
bandplot2 = ListLinePlot[banddata,Frame->True,PlotStyle->Blue,FrameLabel->{"\!\(\*SubscriptBox[\(k\), \(x\)]\) (1/\[Angstrom])","Energy (eV)"},FrameStyle->Directive[Black],
LabelStyle->{FontFamily->"Times New Roman",15},ImageSize->250,AspectRatio->1,PlotLabel->"a = "<>ToString[N@a]<>"\[Angstrom]",PlotRange->{-1,1.5}]


Show[bandplot1,bandplot2]


(*Fourier transform*)
Clear[htr, ftham]
Clear[a, kx,ky,kz,ry,rz];
htr[a_,kx_,ky_,kz_]:=Chop[Simplify[Hdis[a,kx,ky,kz],Element[{kx,ky,kz},Reals]]];
ftham[a_,kx_,ry_,rz_]:= FullSimplify[N[ExpToTrig[FourierTransform[htr[a, kx,ky/a,kz/a]/N[2Pi],{ky,kz},{ry,rz}]]]]

(*Convert Hamiltonian to string*)
hamString=ToString[InputForm[ftham[a, kx,ry,rz]]]


(*Apply all replacements*)
pyString=hamString//StringReplace[{
"DiracDelta"->"dd",
"Sin"->"np.sin","Cos"->"np.cos",
RegularExpression["\\bI\\b"]->"1j",
RegularExpression["([0-9\\.]+)\\s+(np\\.sin)"]->"$1*$2",
RegularExpression["([0-9\\.]+)\\s+(np\\.cos)"]->"$1*$2",
RegularExpression["(np\\.(sin|cos)\\()([^\\)]*?)\\s+([^\\)]*?)(\\))"]:>"$1"<>StringReplace["$3 $4"," "->" * "]<>"$5",
") ("->")*(",
"["->"(","]"->")",
"{"->"[","}"->"]",
"^"->"**"}];

(*(*Export to python file*)Export["hamiltonian.py",pyString,"Text"];*)
(*Display result*)
(*Print["Hamiltonian in Mathematica for N = "<>ToString[NN]<>" :  "]
Print[ftham[kx,ry,rz]]
Print[" "]*)
Print["Hamiltonian in Python for N = "<>ToString[NN]<>" :  "]
Print["def gso(a, kx, ry, rz):
	return np.array(",pyString,")"]


(*Convert Hamiltonian to string*)
hamString=ToString[InputForm[D[ftham[a, kx,ry,rz],kx]]];

(*Apply all replacements*)
pyString=hamString//StringReplace[{
"DiracDelta"->"dd",
"Sin"->"np.sin","Cos"->"np.cos",
RegularExpression["\\bI\\b"]->"1j",
RegularExpression["([0-9\\.]+)\\s+(np\\.sin)"]->"$1*$2",
RegularExpression["([0-9\\.]+)\\s+(np\\.cos)"]->"$1*$2",
RegularExpression["(np\\.(sin|cos)\\()([^\\)]*?)\\s+([^\\)]*?)(\\))"]:>"$1"<>StringReplace["$3 $4"," "->" * "]<>"$5",
") ("->")*(",
"["->"(","]"->")",
"{"->"[","}"->"]",
"^"->"**"}];

(*(*Export to python file*)Export["hamiltonian.py",pyString,"Text"];*)
(*Display result*)
Print["derivative of the Hamiltonian in Python for N = "<>ToString[NN]<>" :  "]
Print[D[ftham[kx,ry,rz],kx]]
Print[" "]
Print["derivative of the Hamiltonian in Python for N = "<>ToString[NN]<>" :  "]
Print["def dgso(a, kx, ry, rz):
	return np.array(",pyString,")"]


(*Convert Hamiltonian to string*)
hamString=ToString[InputForm[D[D[ftham[a,kx,ry,rz],kx],kx]]];

(*Apply all replacements*)
pyString=hamString//StringReplace[{
"DiracDelta"->"dd",
"Sin"->"np.sin","Cos"->"np.cos",
RegularExpression["\\bI\\b"]->"1j",
RegularExpression["([0-9\\.]+)\\s+(np\\.sin)"]->"$1*$2",
RegularExpression["([0-9\\.]+)\\s+(np\\.cos)"]->"$1*$2",
RegularExpression["(np\\.(sin|cos)\\()([^\\)]*?)\\s+([^\\)]*?)(\\))"]:>"$1"<>StringReplace["$3 $4"," "->" * "]<>"$5",
") ("->")*(",
"["->"(","]"->")",
"{"->"[","}"->"]",
"^"->"**"}];

(*(*Export to python file*)Export["hamiltonian.py",pyString,"Text"];*)
(*Display result*)
Print["double derivative of the Hamiltonian in Mathematica for a = "<>ToString[a]<>" :  "]
Print[D[D[ftham[kx,ry,rz],kx],kx]v]
Print[" "]
Print["double derivative of the Hamiltonian in Python for N = "<>ToString[NN]<>" :  "]
Print["def ddgso(a, kx, ry, rz):
	return np.array(",pyString,")"]


(*3D*)
(*Fourier transform*)
Clear[htr, ftham]
Clear[kx,ky,kz,ry,rz];
htr[kx_,ky_,kz_]:=Chop[Simplify[Hdis[kx,ky,kz],Element[{kx,ky,kz},Reals]]];
ftham[rx_,ry_,rz_]:= FullSimplify[N[ExpToTrig[FourierTransform[htr[kx/a,ky/a,kz/a]/N[(2Pi)^(3/2)],{kx,ky,kz},{rx,ry,rz}]]]]

(*Convert Hamiltonian to string*)
hamString=ToString[InputForm[ftham[rx,ry,rz]]];

(*Apply all replacements*)
pyString=hamString//StringReplace[{
"DiracDelta"->"dd",
"Sin"->"np.sin","Cos"->"np.cos",
RegularExpression["\\bI\\b"]->"1j",
RegularExpression["([0-9\\.]+)\\s+(np\\.sin)"]->"$1*$2",
RegularExpression["([0-9\\.]+)\\s+(np\\.cos)"]->"$1*$2",
RegularExpression["(np\\.(sin|cos)\\()([^\\)]*?)\\s+([^\\)]*?)(\\))"]:>"$1"<>StringReplace["$3 $4"," "->" * "]<>"$5",
") ("->")*(",
"["->"(","]"->")",
"{"->"[","}"->"]"}];

(*(*Export to python file*)Export["hamiltonian.py",pyString,"Text"];*)
(*Display result*)
Print["Hamiltonian in Mathematica for N = "<>ToString[NN]<>" :  "]
Print[ftham[rx,ry,rz]]
Print[" "]
Print["Hamiltonian in Python for N = "<>ToString[NN]<>" :  "]
Print["def gso_"<>ToString[NN]<>"(kx, ry, rz):
	return np.array(",pyString,")"]
	
Exit[];

