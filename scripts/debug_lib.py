try:
    import Lib3MF as L
    print("Imported Lib3MF")
except ImportError:
    try:
        import lib3mf as L
        print("Imported lib3mf")
    except ImportError:
        print("Could not import Lib3MF or lib3mf")
        exit(1)

print("Attributes:")
for x in dir(L):
    print(x)

if hasattr(L, 'BaseMaterialGroup'):
    print(f"\nBaseMaterialGroup.AddMaterial: {L.BaseMaterialGroup.AddMaterial}")
    # Try to inspect argtypes if easily accessible?
    # Usually hidden in a wrapper.

