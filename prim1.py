#26.02.24
from dolfin import *

mesh1 = Mesh('prim11.xml')
domains1 = MeshFunction('size_t', mesh1, 'prim11_physical_region.xml')
boundaries1 = MeshFunction ('size_t', mesh1 , 'prim11_facet_region.xml')


mesh2 = Mesh('prim12.xml')
domains2 = MeshFunction('size_t', mesh2, 'prim12_physical_region.xml')
boundaries2 = MeshFunction ('size_t', mesh2 , 'prim12_facet_region.xml')


mesh3 = Mesh('prim13.xml')
domains3 = MeshFunction('size_t', mesh3, 'prim13_physical_region.xml')
boundaries3 = MeshFunction ('size_t', mesh3 , 'prim13_facet_region.xml')

 
mesh4 = Mesh('prim14.xml')
domains4 = MeshFunction('size_t', mesh4, 'prim14_physical_region.xml')
boundaries4 = MeshFunction ('size_t', mesh4 , 'prim14_facet_region.xml')


meshEt = Mesh('prim15.xml')
domainsEt = MeshFunction('size_t', meshEt, 'prim15_physical_region.xml')
boundariesEt = MeshFunction ('size_t', meshEt , 'prim15_facet_region.xml')

VEt = FunctionSpace (meshEt , 'Lagrange' , 1)

dxEt = Measure('dx')(subdomain_data =domainsEt)
dsEt = Measure('ds')(subdomain_data = boundariesEt)

V1 = FunctionSpace (mesh1 , 'Lagrange' , 1)

dx1 = Measure('dx')(subdomain_data =domains1)
ds1 = Measure('ds')(subdomain_data = boundaries1)

V2 = FunctionSpace (mesh2 , 'Lagrange' , 1)

dx2 = Measure('dx')(subdomain_data =domains2)
ds2 = Measure('ds')(subdomain_data = boundaries2)

V3 = FunctionSpace (mesh3 , 'Lagrange' , 1)

dx3 = Measure('dx')(subdomain_data =domains3)
ds3 = Measure('ds')(subdomain_data = boundaries3)

V4 = FunctionSpace (mesh4 , 'Lagrange' , 1)

dx4 = Measure('dx')(subdomain_data =domains4)
ds4 = Measure('ds')(subdomain_data = boundaries4)

f = Constant(1.0)
g = Constant(0.0)

u0_val = Constant(0.0)
p0 = Constant(1000.0)

u0Et = interpolate(u0_val, VEt)
u01 = interpolate(u0_val, V1)
u02 = interpolate(u0_val, V2)
u03 = interpolate(u0_val, V3)
u04 = interpolate(u0_val, V4)

bc1Et = DirichletBC(VEt, g , boundariesEt , 8)
bc11 = DirichletBC(V1, g , boundaries1 , 8)
bc12 = DirichletBC(V2, g , boundaries2 , 8)
bc13 = DirichletBC(V3, g , boundaries3 , 8)
bc14 = DirichletBC(V4, g , boundaries4 , 8)

bc2Et = DirichletBC(VEt, g , boundariesEt , 11)
bc21 = DirichletBC(V1, g , boundaries1 , 11)
bc22 = DirichletBC(V2, g , boundaries2 , 11)
bc23 = DirichletBC(V3, g , boundaries3 , 11)
bc24 = DirichletBC(V4, g , boundaries4 , 11)

bcsEt = [bc1Et, bc2Et]
bcs1 = [bc11, bc21]
bcs2 = [bc12, bc22]
bcs3 = [bc13, bc23]
bcs4 = [bc14, bc24]

k = Constant(10)

T=6.0
N = 100
tau = T/N

info("Solving MeshEt")
uEt = TrialFunction(VEt)
vEt = TestFunction(VEt)
a = (1/tau)*uEt*vEt*dxEt + k*inner(grad(uEt), grad(vEt)) * dxEt
L = f * vEt * dxEt +(1/tau)*u0Et*vEt*dxEt + p0*vEt*dsEt(10) + p0*vEt*dsEt(9)

uEt = Function(VEt)
file = File('./results/time_depEt.pvd')
t=0
while t<T:
	t+=tau
	solve(a == L, uEt, bcsEt)
	file << uEt
	u0Et.assign(uEt)


info("Solving mesh1")
u1 = TrialFunction(V1)
v1 = TestFunction(V1)
a1 = (1/tau)*u1*v1*dx1 + k*inner(grad(u1), grad(v1)) * dx1
L1 = f * v1 * dx1 + (1/tau)*u01*v1*dx1 + p0*v1*ds1(10) + p0*v1*ds1(9)
u1 = Function(V1)
file1 = File('./results/time_dep1.pvd')
t=0
while t<T:
        t+=tau
        solve(a1 == L1, u1, bcs1)
        file1 << u1
        u01.assign(u1)

info("Solving mesh2")
u2 = TrialFunction(V2)
v2 = TestFunction(V2)
a2 = (1/tau)*u2*v2*dx2 + k*inner(grad(u2), grad(v2)) * dx2
L2 = f * v2 * dx2+ (1/tau)*u02*v2*dx2 + p0*v2*ds2(10) + p0*v2*ds2(9)
u2 = Function(V2)
file2 = File('./results/time_dep2.pvd')
t=0
while t<T:
        t+=tau
        solve(a2 == L2, u2, bcs2)
        file2 << u2
        u02.assign(u2)

info("Solving mesh3")
u3 = TrialFunction(V3)
v3 = TestFunction(V3)
a3 = (1/tau)*u3*v3*dx3 + k*inner(grad(u3), grad(v3)) * dx3
L3 = f * v3 * dx3 + (1/tau)*u03*v3*dx3 + p0*v3*ds3(10) + p0*v3*ds3(9)
u3 = Function(V3)
file3 = File('./results/time_dep3.pvd')
t=0
while t<T:
        t+=tau
	solve(a3 == L3, u3, bcs3)
	file3 << u3
        u03.assign(u3)

info("Solving mesh4")
u4 = TrialFunction(V4)
v4 = TestFunction(V4)
a4 = (1/tau)*u4*v4*dx4 + k*inner(grad(u4), grad(v4)) * dx4
L4 = f * v4 * dx4 + (1/tau)*u04*v4*dx4 + p0*v4*ds4(10) + p0*v4*ds4(9)
u4 = Function(V4)
file4 = File('./results/time_dep4.pvd')
t=0
while t<T:
        t+=tau
        solve(a4 == L4, u4, bcs4)
        file4 << u4
        u04.assign(u4)
        
u_e = interpolate(u2, VEt)
E1 = (u_e - uEt)*dx
E1_error = abs(assemble(E1))*100
E2_t = inner(uEt-u_e,uEt-u_e)*dx
E2_b = inner(uEt,uEt)*dx
E2 = sqrt(abs(assemble(E2_t))/abs(assemble(E2_b)))*100
E3_t = inner(grad(uEt-u_e),grad(uEt-u_e))*dx
E3_b = inner(grad(uEt),grad(uEt))*dx
E3 = sqrt(abs(assemble(E3_t))/abs(assemble(E3_b)))*100
print("abs error = " + str(E1_error) + "%\nerror L2 = " + str(E2) + "%\nerror H1 = " + str(E3) + "%")
