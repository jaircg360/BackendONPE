"""
Script de prueba de conexión con Supabase
Ejecutar: python test_connection.py
"""

from database import get_supabase
from tabulate import tabulate

def test_connection():
    print("=" * 60)
    print("PRUEBA DE CONEXIÓN CON SUPABASE")
    print("=" * 60)
    print()
    
    supabase = get_supabase()
    
    # Test 1: Candidatos
    print("📋 Test 1: Obtener Candidatos")
    print("-" * 60)
    try:
        response = supabase.table('candidates').select('*').execute()
        if response.data:
            print(f"✅ Conexión exitosa!")
            print(f"📊 Candidatos encontrados: {len(response.data)}")
            
            # Mostrar tabla de candidatos
            candidates_data = [[c['name'], c['party'], c['is_active']] for c in response.data]
            print(tabulate(candidates_data, headers=['Nombre', 'Partido', 'Activo'], tablefmt='grid'))
        else:
            print("⚠️  No hay candidatos en la base de datos")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Test 2: Datos Electorales
    print("📈 Test 2: Obtener Datos Electorales")
    print("-" * 60)
    try:
        response = supabase.table('electoral_data').select('*').execute()
        if response.data:
            print(f"✅ Datos electorales encontrados: {len(response.data)}")
            
            # Agrupar por año
            years = {}
            for record in response.data:
                year = record.get('election_year')
                if year:
                    years[year] = years.get(year, 0) + 1
            
            years_data = [[year, count] for year, count in sorted(years.items())]
            print(tabulate(years_data, headers=['Año Electoral', 'Registros'], tablefmt='grid'))
        else:
            print("⚠️  No hay datos electorales históricos")
            print("   Ejecuta los scripts SQL de poblado de datos")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Test 3: Configuración del Sistema
    print("⚙️  Test 3: Configuración del Sistema")
    print("-" * 60)
    try:
        response = supabase.table('system_config').select('*').execute()
        if response.data:
            print(f"✅ Configuraciones encontradas: {len(response.data)}")
            
            config_data = [[c['config_key'], str(c['config_value'])[:50]] for c in response.data]
            print(tabulate(config_data, headers=['Clave', 'Valor'], tablefmt='grid'))
        else:
            print("⚠️  No hay configuraciones del sistema")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Test 4: Usuarios
    print("👥 Test 4: Usuarios Registrados")
    print("-" * 60)
    try:
        response = supabase.table('profiles').select('*').execute()
        if response.data:
            print(f"✅ Usuarios encontrados: {len(response.data)}")
        else:
            print("⚠️  No hay usuarios registrados")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Test 5: Votos
    print("🗳️  Test 5: Votos Registrados")
    print("-" * 60)
    try:
        response = supabase.table('votes').select('*', count='exact').execute()
        print(f"✅ Total de votos: {response.count or 0}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Test 6: Tablas de Log
    print("📝 Test 6: Tablas de Logs")
    print("-" * 60)
    log_tables = [
        ('user_activity_log', 'Actividad de Usuarios'),
        ('data_change_log', 'Cambios en Datos'),
        ('voting_log', 'Log de Votación'),
        ('ml_processing_log', 'Procesamiento ML'),
        ('data_cleaning_log', 'Limpieza de Datos'),
        ('system_error_log', 'Errores del Sistema')
    ]
    
    log_counts = []
    for table_name, description in log_tables:
        try:
            response = supabase.table(table_name).select('*', count='exact').execute()
            count = response.count or 0
            log_counts.append([description, count])
        except Exception as e:
            log_counts.append([description, f"Error: {str(e)[:30]}"])
    
    print(tabulate(log_counts, headers=['Tabla de Log', 'Registros'], tablefmt='grid'))
    
    print()
    print("=" * 60)
    print("✅ PRUEBAS COMPLETADAS")
    print("=" * 60)
    print()
    print("📌 PRÓXIMOS PASOS:")
    print("   1. Si hay pocos datos electorales, ejecuta los scripts SQL")
    print("   2. Registra un usuario desde el frontend")
    print("   3. Asigna rol admin desde SQL Editor")
    print("   4. Inicia el backend: python main.py")
    print("   5. Inicia el frontend: npm run dev")
    print()

if __name__ == "__main__":
    try:
        test_connection()
    except Exception as e:
        print(f"\n❌ ERROR FATAL: {e}")
        print("\nVerifica que:")
        print("  - El archivo .env existe y está configurado")
        print("  - Las credenciales de Supabase son correctas")
        print("  - El proyecto de Supabase está activo")

