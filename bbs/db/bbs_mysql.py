from django.db import connection

def get_bbs_with_rownum(offset=0, limit=20):

    print(f" DB Connection {connection.settings_dict}")


    with connection.cursor() as cursor:
        cursor.execute(f"""
            SELECT
                ROW_NUMBER() OVER (ORDER BY group_id DESC, depth ASC, created_at ASC) AS rownum,
                id, type, title, content, writer, created_at, updated_at, is_deleted, group_id, parent_id, depth
            FROM bbs_bbs
            WHERE is_deleted = 0
            ORDER BY group_id DESC, depth ASC, created_at ASC
            LIMIT %s OFFSET %s
        """, [limit, offset])
        columns = [col[0] for col in cursor.description]
        results = [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]
    return results

def get_total_bbs_count():
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(*) FROM bbs_bbs
            WHERE is_deleted = FALSE
        """)
        row = cursor.fetchone()
    return row[0] if row else 0
