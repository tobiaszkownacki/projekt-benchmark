-- The original development seed used a Polish display name. Update only that
-- known seeded account and value; a user-selected name must remain untouched.
UPDATE users
   SET display_name = 'System Administrator'
 WHERE email = 'admin@benchmark.pw.edu.pl'
   AND display_name = 'Administrator systemu';
